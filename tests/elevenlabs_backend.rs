//! ElevenLabs client against a fake HTTP/1.1 server (plain tokio sockets, hand-written responses).
#![cfg(feature = "elevenlabs")]

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mcp_sfxgen::backend::elevenlabs::{
    DEFAULT_BASE_URL, DEFAULT_TIMEOUT, ElevenLabsClient, ElevenLabsError, PcmFormat, SfxRequest,
};
use mcp_sfxgen::secrets::ApiKey;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;

const KEY: &str = "sk_0123456789abcdef0123456789abcdef";

enum Reply {
    Respond {
        status: u16,
        headers: Vec<(&'static str, String)>,
        body: Vec<u8>,
    },
    /// Read the request, then never answer.
    Hang,
}

impl Reply {
    fn json(status: u16, body: &str) -> Reply {
        Reply::Respond {
            status,
            headers: vec![("content-type", "application/json".into())],
            body: body.as_bytes().to_vec(),
        }
    }

    fn json_with(status: u16, headers: Vec<(&'static str, String)>, body: &str) -> Reply {
        let mut all = vec![("content-type", "application/json".to_string())];
        all.extend(headers);
        Reply::Respond {
            status,
            headers: all,
            body: body.as_bytes().to_vec(),
        }
    }

    fn audio(bytes: usize, credits: u32) -> Reply {
        Reply::Respond {
            status: 200,
            headers: vec![
                ("content-type", "audio/pcm".into()),
                ("character-cost", credits.to_string()),
                ("request-id", "req-abc".into()),
            ],
            body: pcm(bytes),
        }
    }
}

fn pcm(bytes: usize) -> Vec<u8> {
    (0..bytes).map(|i| (i % 251) as u8).collect()
}

#[derive(Debug, Clone)]
struct Recorded {
    request_line: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

struct FakeServer {
    addr: SocketAddr,
    recorded: Arc<Mutex<Vec<Recorded>>>,
    hung_closed: Arc<AtomicUsize>,
}

impl FakeServer {
    async fn start(script: Vec<Reply>) -> FakeServer {
        capture();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let script = Arc::new(Mutex::new(VecDeque::from(script)));
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let hung_closed = Arc::new(AtomicUsize::new(0));
        let (s, r, h) = (script.clone(), recorded.clone(), hung_closed.clone());
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(serve(stream, s.clone(), r.clone(), h.clone()));
            }
        });
        FakeServer {
            addr,
            recorded,
            hung_closed,
        }
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn requests(&self) -> Vec<Recorded> {
        self.recorded.lock().unwrap().clone()
    }

    fn client(&self, timeout: Duration) -> ElevenLabsClient {
        // Short retry delays; jitter still applies.
        ElevenLabsClient::new(self.url(), timeout)
            .unwrap()
            .with_retry_delays([Duration::from_millis(20), Duration::from_millis(40)])
    }
}

async fn serve(
    mut stream: TcpStream,
    script: Arc<Mutex<VecDeque<Reply>>>,
    recorded: Arc<Mutex<Vec<Recorded>>>,
    hung_closed: Arc<AtomicUsize>,
) {
    let mut buf = Vec::new();
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        let mut chunk = [0u8; 4096];
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("").to_string();
    let headers: HashMap<String, String> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    while buf.len() < header_end + length {
        let mut chunk = [0u8; 4096];
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    recorded.lock().unwrap().push(Recorded {
        request_line,
        headers,
        body: buf[header_end..header_end + length].to_vec(),
    });

    let reply = script.lock().unwrap().pop_front().unwrap_or_else(|| {
        Reply::json(
            500,
            r#"{"detail":{"message":"fake server script exhausted"}}"#,
        )
    });
    match reply {
        Reply::Hang => {
            // Wait for the client to give up; record that it did.
            let mut chunk = [0u8; 64];
            while let Ok(n) = stream.read(&mut chunk).await {
                if n == 0 {
                    break;
                }
            }
            hung_closed.fetch_add(1, Ordering::SeqCst);
        }
        Reply::Respond {
            status,
            headers,
            body,
        } => {
            let reason = match status {
                200 => "OK",
                401 => "Unauthorized",
                402 => "Payment Required",
                422 => "Unprocessable Entity",
                429 => "Too Many Requests",
                500 => "Internal Server Error",
                503 => "Service Unavailable",
                _ => "Status",
            };
            let mut out = format!("HTTP/1.1 {status} {reason}\r\n").into_bytes();
            for (k, v) in headers {
                out.extend_from_slice(format!("{k}: {v}\r\n").as_bytes());
            }
            out.extend_from_slice(
                format!(
                    "content-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            );
            out.extend_from_slice(&body);
            let _ = stream.write_all(&out).await;
            let _ = stream.shutdown().await;
        }
    }
}

fn key() -> ApiKey {
    ApiKey::new(KEY)
}

fn sfx(text: &str, seconds: Option<f64>) -> SfxRequest {
    let mut req = SfxRequest::new(text);
    req.duration_seconds = seconds;
    req
}

const BUSY: &str = r#"{"detail":{"type":"rate_limit","code":"system_busy","message":"We are experiencing high traffic","status":"system_busy","request_id":"r1"}}"#;

#[tokio::test]
async fn success_sends_the_documented_request() {
    let server = FakeServer::start(vec![Reply::audio(176_400, 10)]).await;
    let mut req = sfx("a small horn, two short beeps", Some(1.0));
    req.prompt_influence = 0.5;
    req.loop_ = true;
    let audio = server
        .client(Duration::from_secs(5))
        .generate(&key(), &req, &CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(audio.pcm_s16le.len(), 176_400);
    assert_eq!(audio.pcm_s16le, pcm(176_400));
    assert_eq!((audio.sample_rate, audio.channels), (44_100, 2));
    assert_eq!(audio.credits, Some(10));
    assert_eq!(audio.request_id.as_deref(), Some("req-abc"));
    assert_eq!(audio.attempts, 1);

    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    let r = &requests[0];
    assert_eq!(
        r.request_line,
        "POST /v1/sound-generation?output_format=pcm_44100 HTTP/1.1"
    );
    assert_eq!(r.headers.get("xi-api-key").map(String::as_str), Some(KEY));
    assert!(r.headers["content-type"].starts_with("application/json"));
    let body: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
    assert_eq!(
        body,
        serde_json::json!({
            "text": "a small horn, two short beeps",
            "model_id": "eleven_text_to_sound_v2",
            "duration_seconds": 1.0,
            "prompt_influence": 0.5,
            "loop": true
        })
    );
}

#[tokio::test]
async fn automatic_length_omits_the_duration_and_48k_changes_the_query() {
    let server = FakeServer::start(vec![Reply::audio(960_000, 50)]).await;
    let mut req = sfx("rain", None);
    req.output_format = PcmFormat::Pcm48000;
    let audio = server
        .client(Duration::from_secs(5))
        .generate(&key(), &req, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!((audio.sample_rate, audio.credits), (48_000, Some(50)));
    let r = &server.requests()[0];
    assert_eq!(
        r.request_line,
        "POST /v1/sound-generation?output_format=pcm_48000 HTTP/1.1"
    );
    let body: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
    assert!(body.get("duration_seconds").is_none(), "{body}");
    assert_eq!(body["loop"], false);
    assert_eq!(body["prompt_influence"], 0.3);
}

#[tokio::test]
async fn a_missing_cost_header_gives_none() {
    let server = FakeServer::start(vec![Reply::Respond {
        status: 200,
        headers: vec![("content-type", "audio/pcm".into())],
        body: pcm(8),
    }])
    .await;
    let audio = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!((audio.credits, audio.request_id), (None, None));
}

#[tokio::test]
async fn account_without_user_read_is_a_permission_error() {
    let server = FakeServer::start(vec![Reply::json(
        401,
        r#"{"detail":{"type":"authentication_error","code":"missing_permissions","message":"The API key you used is missing the permission user_read to execute this operation.","status":"missing_permissions","request_id":"r2"}}"#,
    )])
    .await;
    let err = server
        .client(Duration::from_secs(5))
        .account(&key())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ElevenLabsError::Permission { permission, .. } if permission == "user_read"),
        "{err:?}"
    );
    assert_eq!(err.code(), "elevenlabs_permission");
    let requests = server.requests();
    assert_eq!(requests.len(), 1, "401 is never retried");
    assert_eq!(
        requests[0].request_line,
        "GET /v1/user/subscription HTTP/1.1"
    );
    assert_eq!(
        requests[0].headers.get("xi-api-key").map(String::as_str),
        Some(KEY)
    );
}

#[tokio::test]
async fn account_reads_the_subscription() {
    let server = FakeServer::start(vec![Reply::json(
        200,
        r#"{"tier":"free","status":"free","character_count":130,"character_limit":10000,"next_character_count_reset_unix":1793600000,"can_use_instant_voice_cloning":false}"#,
    )])
    .await;
    let info = server
        .client(Duration::from_secs(5))
        .account(&key())
        .await
        .unwrap();
    assert_eq!(info.tier, "free");
    assert_eq!(info.status, "free");
    assert_eq!((info.character_count, info.character_limit), (130, 10_000));
    assert_eq!(info.next_reset_unix, Some(1_793_600_000));

    let server = FakeServer::start(vec![Reply::json(200, "not json")]).await;
    let err = server
        .client(Duration::from_secs(5))
        .account(&key())
        .await
        .unwrap_err();
    assert!(matches!(err, ElevenLabsError::BadResponse(_)), "{err:?}");
}

#[tokio::test]
async fn invalid_key_is_auth_and_not_retried() {
    let server = FakeServer::start(vec![Reply::json(
        401,
        r#"{"detail":{"type":"authentication_error","code":"invalid_api_key","message":"Invalid API key","status":"invalid_api_key"}}"#,
    )])
    .await;
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, ElevenLabsError::Auth { .. }), "{err:?}");
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn payment_required_is_quota_and_not_retried() {
    let server = FakeServer::start(vec![Reply::json(
        402,
        r#"{"detail":{"status":"payment_required","message":"Free users cannot use this."}}"#,
    )])
    .await;
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ElevenLabsError::Quota { message } if message.contains("Free users")),
        "{err:?}"
    );
    assert_eq!(err.code(), "elevenlabs_quota");
    assert_eq!(server.requests().len(), 1);

    let server = FakeServer::start(vec![Reply::json(
        401,
        r#"{"detail":{"status":"quota_exceeded","message":"This request exceeds your quota of 10000. You have 5 credits remaining, while 50 credits are required."}}"#,
    )])
    .await;
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", None), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ElevenLabsError::Quota { message } if message.contains("5 credits")),
        "{err:?}"
    );
}

#[tokio::test]
async fn validation_array_names_the_field() {
    let server = FakeServer::start(vec![Reply::json(
        422,
        r#"{"detail":[{"type":"less_than_equal","loc":["body","duration_seconds"],"msg":"Input should be less than or equal to 30","input":31}]}"#,
    )])
    .await;
    // The request itself is in range, so it reaches the server and the server's 422 is what we see.
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(10.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ElevenLabsError::Validation { field, message }
            if field == "duration_seconds" && message.contains("less than or equal")),
        "{err:?}"
    );
    assert_eq!(err.code(), "elevenlabs_validation");
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn out_of_range_requests_are_refused_before_sending() {
    let server = FakeServer::start(vec![]).await;
    let client = server.client(Duration::from_secs(5));
    let err = client
        .generate(&key(), &sfx("x", Some(31.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ElevenLabsError::Validation { field, .. } if field == "duration_seconds")
    );
    // A key that can't be a header value is also caught locally.
    let err = client
        .generate(
            &ApiKey::new("bad\nkey"),
            &sfx("x", Some(1.0)),
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, ElevenLabsError::Auth { .. }));
    assert_eq!(server.requests().len(), 0);
}

#[tokio::test]
async fn system_busy_then_success_reports_two_attempts_and_only_the_final_cost() {
    let server = FakeServer::start(vec![
        // A failed attempt may even carry a cost header; it must not be counted.
        Reply::json_with(429, vec![("character-cost", "99".into())], BUSY),
        Reply::audio(176_400, 10),
    ])
    .await;
    let audio = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(audio.attempts, 2);
    assert_eq!(audio.credits, Some(10));
    assert_eq!(server.requests().len(), 2);
}

#[tokio::test]
async fn too_many_concurrent_requests_is_returned_without_retry() {
    let server = FakeServer::start(vec![Reply::json(
        429,
        r#"{"detail":{"type":"rate_limit","code":"too_many_concurrent_requests","message":"Too many concurrent requests. Your plan allows 2.","status":"too_many_concurrent_requests"}}"#,
    )])
    .await;
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(err, ElevenLabsError::Concurrency { .. }),
        "{err:?}"
    );
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn server_errors_are_retried_three_times_with_backoff() {
    let server = FakeServer::start(vec![
        Reply::json(500, r#"{"detail":{"message":"boom 1"}}"#),
        Reply::json(503, r#"{"detail":{"message":"boom 2"}}"#),
        Reply::json(500, r#"{"detail":{"message":"boom 3"}}"#),
        Reply::audio(8, 10), // must never be reached
    ])
    .await;
    let client = server
        .client(Duration::from_secs(5))
        .with_retry_delays([Duration::from_millis(200), Duration::from_millis(200)]);
    let started = Instant::now();
    let err = client
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    let elapsed = started.elapsed();
    assert!(
        matches!(&err, ElevenLabsError::Server { status: 500, message } if message == "boom 3"),
        "{err:?}"
    );
    assert_eq!(server.requests().len(), 3);
    // Two sleeps of 200 ms, each jittered to at least 150 ms.
    assert!(elapsed >= Duration::from_millis(290), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(3), "{elapsed:?}");
}

#[tokio::test]
async fn busy_forever_ends_as_busy_after_three_attempts() {
    let server = FakeServer::start(vec![
        Reply::json(429, BUSY),
        Reply::json(429, BUSY),
        Reply::json(429, BUSY),
    ])
    .await;
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, ElevenLabsError::Busy { .. }), "{err:?}");
    assert_eq!(err.code(), "elevenlabs_busy");
    assert_eq!(server.requests().len(), 3);
}

#[tokio::test]
async fn retry_after_is_honoured() {
    // Retry-After: 0 beats a 5 s default delay.
    let server = FakeServer::start(vec![
        Reply::json_with(429, vec![("retry-after", "0".into())], BUSY),
        Reply::audio(8, 10),
    ])
    .await;
    let client = server
        .client(Duration::from_secs(5))
        .with_retry_delays([Duration::from_secs(5), Duration::from_secs(5)]);
    let started = Instant::now();
    let audio = client
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(audio.attempts, 2);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );

    // Retry-After: 1 beats a tiny default delay (and is not jittered down).
    let server = FakeServer::start(vec![
        Reply::json_with(
            503,
            vec![("retry-after", "1".into())],
            r#"{"detail":{"message":"later"}}"#,
        ),
        Reply::audio(8, 10),
    ])
    .await;
    let client = server
        .client(Duration::from_secs(5))
        .with_retry_delays([Duration::from_millis(5), Duration::from_millis(5)]);
    let started = Instant::now();
    let audio = client
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(audio.attempts, 2);
    assert!(
        started.elapsed() >= Duration::from_millis(950),
        "{:?}",
        started.elapsed()
    );

    // An hour is not waited for: the error comes back at once.
    let server = FakeServer::start(vec![Reply::json_with(
        429,
        vec![("retry-after", "3600".into())],
        BUSY,
    )])
    .await;
    let started = Instant::now();
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, ElevenLabsError::Busy { .. }));
    assert_eq!(server.requests().len(), 1);
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[tokio::test]
async fn a_silent_server_times_out_after_three_attempts() {
    let server = FakeServer::start(vec![Reply::Hang, Reply::Hang, Reply::Hang]).await;
    let started = Instant::now();
    let err = server
        .client(Duration::from_millis(300))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(err, ElevenLabsError::Timeout);
    assert_eq!(err.code(), "timeout");
    assert_eq!(server.requests().len(), 3);
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[tokio::test]
async fn connection_refused_is_a_network_error() {
    // A port that was bound a moment ago and is now closed.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let client = ElevenLabsClient::new(format!("http://{addr}"), Duration::from_secs(5))
        .unwrap()
        .with_retry_delays([Duration::from_millis(10), Duration::from_millis(10)]);
    let err = client
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, ElevenLabsError::Network(_)), "{err:?}");
    assert_eq!(err.code(), "network");
}

#[tokio::test]
async fn bad_audio_is_never_retried() {
    // Odd length: not a whole number of 4-byte frames.
    let server = FakeServer::start(vec![Reply::audio(176_401, 10)]).await;
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, ElevenLabsError::BadAudio(_)), "{err:?}");
    assert_eq!(
        server.requests().len(),
        1,
        "credits were spent; no second call"
    );

    // 2 bytes is a multiple of 2 but not of 4.
    let server = FakeServer::start(vec![Reply::audio(2, 10)]).await;
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, ElevenLabsError::BadAudio(_)));

    // Empty.
    let server = FakeServer::start(vec![Reply::audio(0, 0)]).await;
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, ElevenLabsError::BadAudio(_)));

    // A 200 that is JSON, not audio.
    let server = FakeServer::start(vec![Reply::json(
        200,
        r#"{"hello":"world","pad":"1234567"}"#,
    )])
    .await;
    let err = server
        .client(Duration::from_secs(5))
        .generate(&key(), &sfx("x", Some(1.0)), &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ElevenLabsError::BadAudio(m) if m.contains("json")),
        "{err:?}"
    );
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn cancelling_mid_request_aborts_it() {
    let server = FakeServer::start(vec![Reply::Hang]).await;
    let client = server.client(Duration::from_secs(30));
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        trigger.cancel();
    });
    let started = Instant::now();
    let err = client
        .generate(&key(), &sfx("x", Some(1.0)), &cancel)
        .await
        .unwrap_err();
    assert_eq!(err, ElevenLabsError::Cancelled);
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_eq!(server.requests().len(), 1, "cancelled, not retried");
    // The connection was really dropped: the server saw EOF.
    for _ in 0..40 {
        if server.hung_closed.load(Ordering::SeqCst) == 1 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("the server never saw the connection close");
}

#[tokio::test]
async fn cancelling_during_backoff_and_before_starting() {
    let server = FakeServer::start(vec![Reply::json(500, "{}"), Reply::audio(8, 10)]).await;
    let client = server
        .client(Duration::from_secs(5))
        .with_retry_delays([Duration::from_secs(10), Duration::from_secs(10)]);
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        trigger.cancel();
    });
    let started = Instant::now();
    let err = client
        .generate(&key(), &sfx("x", Some(1.0)), &cancel)
        .await
        .unwrap_err();
    assert_eq!(err, ElevenLabsError::Cancelled);
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_eq!(server.requests().len(), 1);

    // Already cancelled: nothing is sent.
    let before = server.requests().len();
    let err = client
        .generate(&key(), &sfx("x", Some(1.0)), &cancel)
        .await
        .unwrap_err();
    assert_eq!(err, ElevenLabsError::Cancelled);
    assert_eq!(server.requests().len(), before);
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// One global capture for the whole test binary, started by the first `FakeServer`: a thread-local subscriber raced with
/// the other tests' threads over tracing's callsite cache. Every test's log lines land here, so the check below also
/// proves that no other test logs a key.
fn capture() -> &'static Capture {
    static CAPTURE: std::sync::OnceLock<Capture> = std::sync::OnceLock::new();
    CAPTURE.get_or_init(|| {
        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
        capture
    })
}

/// R9: no error string, Debug output or tracing line contains the key, even when the server echoes it back.
#[tokio::test]
async fn the_key_never_appears_in_errors_debug_output_or_logs() {
    let capture = capture().clone();

    let sk_key = ApiKey::new(KEY);
    let plain_key = ApiKey::new("plainsecret-9f8e7d6c5b4a");
    let mut shown = Vec::new();
    let mut note = |label: &str, text: String| shown.push(format!("{label}: {text}"));

    let echo = |k: &str| {
        format!(
            r#"{{"detail":{{"status":"invalid_api_key","message":"Invalid API key {k}. xi-api-key: {k}"}}}}"#
        )
    };
    // The server echoes each key in a 401, a 422 field, a 500 body and a 200 that isn't audio.
    let script = vec![
        Reply::json(401, &echo(KEY)),
        Reply::json(
            422,
            &format!(r#"{{"detail":[{{"loc":["body","{KEY}"],"msg":"bad {KEY}"}}]}}"#),
        ),
        Reply::json(500, &echo(KEY)),
        Reply::json(500, &echo(KEY)),
        Reply::json(500, &echo(KEY)),
        Reply::json(401, &echo("plainsecret-9f8e7d6c5b4a")),
        Reply::json(200, &echo(KEY)),
    ];
    let server = FakeServer::start(script).await;
    let client = server.client(Duration::from_secs(5));
    let cancel = CancellationToken::new();

    for (label, k) in [
        ("401", &sk_key),
        ("422", &sk_key),
        ("500", &sk_key),
        ("401-plain", &plain_key),
        ("200-json", &sk_key),
    ] {
        let err = client
            .generate(k, &sfx("x", Some(1.0)), &cancel)
            .await
            .unwrap_err();
        note(
            label,
            format!("{err} | {err:?} | {} | {}", err.code(), err.fix()),
        );
    }
    // The permission path too.
    let server2 = FakeServer::start(vec![Reply::json(
        401,
        &format!(r#"{{"detail":{{"status":"missing_permissions","message":"key {KEY} is missing the permission user_read"}}}}"#),
    )])
    .await;
    let err = server2
        .client(Duration::from_secs(5))
        .account(&sk_key)
        .await
        .unwrap_err();
    note("perm", format!("{err} | {err:?} | {}", err.fix()));
    // And the key's own formatting, and a connection error to a closed port.
    note(
        "key",
        format!("{sk_key} {sk_key:?} {plain_key} {plain_key:?}"),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let closed = listener.local_addr().unwrap();
    drop(listener);
    let err = ElevenLabsClient::new(format!("http://{closed}"), Duration::from_secs(2))
        .unwrap()
        .with_retry_delays([Duration::from_millis(5), Duration::from_millis(5)])
        .generate(&sk_key, &sfx("x", Some(1.0)), &cancel)
        .await
        .unwrap_err();
    note("refused", format!("{err} | {err:?}"));

    let logs = String::from_utf8_lossy(&capture.0.lock().unwrap()).to_string();
    assert!(
        logs.contains("ElevenLabs attempt failed"),
        "the capture should have caught our own debug lines, got {} bytes",
        logs.len()
    );
    let everything = format!("{}\n{logs}", shown.join("\n"));
    for secret in [
        KEY,
        "0123456789abcdef0123456789abcdef",
        "plainsecret-9f8e7d6c5b4a",
        "9f8e7d6c5b4a",
    ] {
        assert!(
            !everything.contains(secret),
            "found `{secret}` in:\n{everything}"
        );
    }
    assert!(
        everything.contains("sk_<redacted>"),
        "the masks should be visible:\n{shown:?}"
    );
}

#[test]
fn defaults_are_the_documented_ones() {
    assert_eq!(DEFAULT_BASE_URL, "https://api.elevenlabs.io");
    assert_eq!(DEFAULT_TIMEOUT, Duration::from_secs(60));
    let client = ElevenLabsClient::new("http://localhost:1/", DEFAULT_TIMEOUT).unwrap();
    assert_eq!(
        client.base_url(),
        "http://localhost:1",
        "a trailing slash is dropped"
    );
}

// ---------------------------------------------------------------------------------------------------------------------
// The `SfxBackend` adapter, against the same fake server.

mod adapter {
    use super::*;
    use mcp_sfxgen::backend::elevenlabs::ElevenLabsBackend;
    use mcp_sfxgen::backend::{
        AccountError, BackendError, BackendRequest, ProgressSink, Runs, SfxBackend,
    };
    use mcp_sfxgen::config::{ElevenLabsConfig, MODELS_EXAMPLE};
    use mcp_sfxgen::ledger::Ledger;
    use mcp_sfxgen::registry::{ModelRegistry, ModelSpec};
    use mcp_sfxgen::secrets::SecretError;

    fn model() -> ModelSpec {
        ModelRegistry::from_toml(MODELS_EXAMPLE)
            .unwrap()
            .get("elevenlabs-sfx-v2")
            .unwrap()
            .clone()
    }

    fn config(concurrency: usize) -> ElevenLabsConfig {
        ElevenLabsConfig {
            api_key_env: "UNUSED_IN_TESTS".into(),
            keychain_service: "unused-in-tests".into(),
            plan: "free".into(),
            concurrency,
            per_call_credit_limit: 300,
            daily_credit_limit: 2000,
            credits_per_second: 10,
            credits_auto_duration: 50,
            timeout_s: 5,
        }
    }

    fn backend(server: &FakeServer, concurrency: usize) -> ElevenLabsBackend {
        ElevenLabsBackend::with_base_url(config(concurrency), server.url())
            .unwrap()
            .with_key_provider(|| Ok(Some(ApiKey::new(KEY))))
            .with_retry_delays([Duration::from_millis(10), Duration::from_millis(20)])
            .with_concurrency_wait(Duration::from_millis(20))
    }

    fn request(variations: u32) -> BackendRequest {
        BackendRequest {
            prompt: "a small horn".into(),
            duration_s: Some(1.0),
            variations,
            sample_rate: 44_100,
            ..Default::default()
        }
    }

    async fn run(
        backend: &ElevenLabsBackend,
        req: &BackendRequest,
    ) -> Result<Vec<mcp_sfxgen::backend::RawClip>, BackendError> {
        backend
            .generate(
                &model(),
                req,
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
    }

    #[tokio::test(flavor = "current_thread")]
    async fn variations_come_back_in_order_with_credits_and_defaults_from_the_model() {
        let server =
            FakeServer::start(vec![Reply::audio(176_400, 10), Reply::audio(176_400, 10)]).await;
        let b = backend(&server, 2);
        assert_eq!(
            (b.id(), b.runs(), b.provider()),
            ("elevenlabs", Runs::Cloud, Some("elevenlabs"))
        );
        let clips = run(&b, &request(2)).await.unwrap();
        assert_eq!(clips.len(), 2);
        for (i, clip) in clips.iter().enumerate() {
            assert_eq!((clip.sample_rate, clip.channels), (44_100, 2));
            assert_eq!(clip.frames(), 44_100);
            assert_eq!(clip.seed, None);
            assert_eq!(clip.credits, Some(10));
            assert_eq!(clip.source_format, "pcm_44100");
            assert_eq!(clip.details["batch"]["index"], i);
            assert_eq!(clip.details["request_id"], "req-abc");
            // s16le -> f32: the fake's byte pattern is deterministic, so the first sample is checkable.
            let expected = i16::from_le_bytes([pcm(4)[0], pcm(4)[1]]) as f32 / 32768.0;
            assert!((clip.samples[0] - expected).abs() < 1e-9);
        }
        let requests = server.requests();
        assert_eq!(requests.len(), 2);
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["prompt_influence"], 0.3, "the model's default");
        assert_eq!(body["model_id"], "eleven_text_to_sound_v2");
        assert_eq!(body["duration_seconds"], 1.0);
        assert_eq!(body["loop"], false);
        assert!(requests[0].request_line.contains("output_format=pcm_44100"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn loop_influence_and_48k_are_passed_through() {
        let server = FakeServer::start(vec![Reply::audio(192_000, 10)]).await;
        let b = backend(&server, 2);
        let mut req = request(1);
        req.loop_ = true;
        req.prompt_influence = Some(0.9);
        req.sample_rate = 48_000;
        let clips = run(&b, &req).await.unwrap();
        assert_eq!(clips[0].sample_rate, 48_000);
        assert_eq!(clips[0].source_format, "pcm_48000");
        let r = &server.requests()[0];
        assert!(r.request_line.contains("output_format=pcm_48000"));
        let body: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(
            (body["loop"].clone(), body["prompt_influence"].clone()),
            (true.into(), 0.9.into())
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn an_automatic_length_sends_no_duration() {
        let server = FakeServer::start(vec![Reply::audio(88_200, 50)]).await;
        let mut req = request(1);
        req.duration_s = None;
        let clips = run(&backend(&server, 2), &req).await.unwrap();
        assert_eq!(clips[0].credits, Some(50));
        let body: serde_json::Value = serde_json::from_slice(&server.requests()[0].body).unwrap();
        assert!(body.get("duration_seconds").is_none(), "{body}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn too_many_concurrent_requests_waits_and_retries() {
        let server = FakeServer::start(vec![
            Reply::json(
                429,
                r#"{"detail":{"type":"rate_limit","code":"too_many_concurrent_requests","message":"slow down","status":"too_many_concurrent_requests"}}"#,
            ),
            Reply::audio(176_400, 10),
        ])
        .await;
        let clips = run(&backend(&server, 1), &request(1)).await.unwrap();
        assert_eq!(clips.len(), 1);
        assert_eq!(server.requests().len(), 2, "one refused, one served");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_refused_request_is_explained_and_scrubbed() {
        let server = FakeServer::start(vec![Reply::json(
            402,
            &format!(
                r#"{{"detail":{{"type":"payment","code":"quota_exceeded","message":"You have 4 credits left (key {KEY})","status":"quota_exceeded"}}}}"#
            ),
        )])
        .await;
        let err = run(&backend(&server, 2), &request(1)).await.unwrap_err();
        let BackendError::Failed(f) = err else {
            panic!()
        };
        assert_eq!(f.kind, "elevenlabs_quota");
        assert_eq!(f.stage, "request");
        assert!(f.hint.contains("account_status"), "{}", f.hint);
        assert!(!f.message.contains(KEY) && !f.hint.contains(KEY), "{f:?}");
        assert!(f.message.contains("sk_<redacted>"), "{}", f.message);
        let e = BackendError::Failed(f).into_error("elevenlabs");
        assert_eq!(e.code(), "elevenlabs_quota");
        assert!(!e.to_string().contains(KEY));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn credits_of_discarded_siblings_go_into_the_ledger() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(dir.path().join("credits.jsonl"));
        let server = FakeServer::start(vec![
            Reply::audio(176_400, 10),
            Reply::json(
                402,
                r#"{"detail":{"type":"payment","code":"quota_exceeded","message":"out of credits","status":"quota_exceeded"}}"#,
            ),
        ])
        .await;
        // One slot: the first request finishes (and is paid for) before the second is refused.
        let b = backend(&server, 1).with_ledger(ledger.clone());
        let err = run(&b, &request(2)).await.unwrap_err();
        assert!(
            matches!(&err, BackendError::Failed(f) if f.kind == "elevenlabs_quota"),
            "{err:?}"
        );
        assert_eq!(ledger.today_total().unwrap(), 10);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_missing_key_is_unavailable_with_the_hint_and_the_key_is_cached() {
        let server = FakeServer::start(vec![]).await;
        let b = ElevenLabsBackend::with_base_url(config(2), server.url())
            .unwrap()
            .with_key_provider(|| Ok(None));
        let err = run(&b, &request(1)).await.unwrap_err();
        assert!(
            matches!(&err, BackendError::Unavailable { reason } if reason.contains("UNUSED_IN_TESTS")),
            "{err:?}"
        );
        assert!(
            server.requests().is_empty(),
            "nothing is sent without a key"
        );

        let lookups = Arc::new(AtomicUsize::new(0));
        let l = lookups.clone();
        let server =
            FakeServer::start(vec![Reply::audio(176_400, 10), Reply::audio(176_400, 10)]).await;
        let b = ElevenLabsBackend::with_base_url(config(2), server.url())
            .unwrap()
            .with_key_provider(move || {
                l.fetch_add(1, Ordering::SeqCst);
                Ok(Some(ApiKey::new(KEY)))
            });
        run(&b, &request(1)).await.unwrap();
        run(&b, &request(1)).await.unwrap();
        assert_eq!(
            lookups.load(Ordering::SeqCst),
            1,
            "looked up once, reused for 60 s"
        );

        let b = ElevenLabsBackend::with_base_url(config(2), server.url())
            .unwrap()
            .with_key_provider(|| Err(SecretError::Timeout(10)));
        let err = run(&b, &request(1)).await.unwrap_err();
        assert!(
            matches!(&err, BackendError::Failed(f) if f.kind == "credential_lookup"),
            "{err:?}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn bad_ranges_are_refused_before_sending() {
        let server = FakeServer::start(vec![]).await;
        let mut req = request(1);
        req.duration_s = Some(99.0);
        let err = run(&backend(&server, 2), &req).await.unwrap_err();
        assert!(
            matches!(&err, BackendError::Failed(f) if f.kind == "elevenlabs_validation"),
            "{err:?}"
        );
        assert!(server.requests().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelling_stops_the_requests() {
        let server = FakeServer::start(vec![Reply::Hang, Reply::Hang]).await;
        let b = backend(&server, 2);
        let cancel = CancellationToken::new();
        let c = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            c.cancel();
        });
        let started = Instant::now();
        let err = b
            .generate(&model(), &request(2), ProgressSink::noop(), cancel)
            .await
            .unwrap_err();
        assert_eq!(err, BackendError::Cancelled);
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn account_reads_the_plan_and_is_cached() {
        let server = FakeServer::start(vec![Reply::json(
            200,
            r#"{"tier":"free","status":"free","character_count":130,"character_limit":10000,"next_character_count_reset_unix":1793600000}"#,
        )])
        .await;
        let b = backend(&server, 2);
        let snap = b.account().await.unwrap().unwrap();
        assert_eq!(snap.plan, "free");
        assert_eq!(
            (
                snap.credits_used,
                snap.credits_limit,
                snap.credits_remaining
            ),
            (130, 10_000, 9_870)
        );
        assert!(
            snap.resets_at.unwrap().starts_with("2026-"),
            "a unix time in late 2026"
        );
        b.account().await.unwrap().unwrap();
        assert_eq!(
            server.requests().len(),
            1,
            "the second answer came from the cache"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn account_without_user_read_is_not_readable() {
        let server = FakeServer::start(vec![Reply::json(
            401,
            r#"{"detail":{"type":"authentication_error","code":"missing_permissions","message":"The API key you used is missing the permission user_read to execute this operation.","status":"missing_permissions"}}"#,
        )])
        .await;
        let b = backend(&server, 2);
        let result = b.account().await.unwrap();
        let Err(AccountError::NotReadable { reason }) = result else {
            panic!("{result:?}")
        };
        assert!(reason.contains("user_read"), "{reason}");
        assert!(b.account().await.unwrap().is_err());
        assert_eq!(
            server.requests().len(),
            1,
            "a 401 is remembered for the cache period too"
        );
    }
}

/// Spends 10 credits on the real API. Run by hand: `ELEVENLABS_API_KEY=... cargo test -- --ignored live`.
#[tokio::test]
#[ignore = "calls the real ElevenLabs API and spends 10 credits"]
async fn live_one_second_costs_ten_credits() {
    let key = std::env::var("ELEVENLABS_API_KEY").expect("set ELEVENLABS_API_KEY");
    let key = ApiKey::new(key.trim());
    let client = ElevenLabsClient::new(DEFAULT_BASE_URL, DEFAULT_TIMEOUT).unwrap();
    let audio = client
        .generate(
            &key,
            &sfx("a short soft click", Some(1.0)),
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(audio.credits, Some(10));
    assert_eq!(audio.pcm_s16le.len(), 176_400);
}
