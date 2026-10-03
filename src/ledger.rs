//! Credit ledger and cost control (requirements R10, design §4.3 and §7).
//!
//! Every ElevenLabs call's `character-cost` header is appended to a JSON-lines file, so spend is known even when
//! the key can't read the account. Totals are summed by *local* date. The estimate and the limit checks run before
//! anything is sent.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Datelike, Local, SecondsFormat};
use serde::{Deserialize, Serialize};

/// One spend. `ts` is written as RFC 3339 with the local offset.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub ts: DateTime<Local>,
    pub model: String,
    pub credits: u32,
    pub duration_s: Option<f64>,
    pub file: Option<String>,
    pub request_id: Option<String>,
}

impl Entry {
    /// An entry stamped now, with no optional fields.
    pub fn now(model: impl Into<String>, credits: u32) -> Self {
        Entry {
            ts: Local::now(),
            model: model.into(),
            credits,
            duration_s: None,
            file: None,
            request_id: None,
        }
    }
}

/// The on-disk shape of a line.
#[derive(Serialize, Deserialize)]
struct Row {
    ts: String,
    model: String,
    credits: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    duration_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum LedgerError {
    #[error("ledger {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("ledger entry could not be encoded: {0}")]
    Encode(String),
}

/// Today's and this month's spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Totals {
    pub today: u64,
    pub month: u64,
}

/// A JSON-lines credit ledger. Cheap to clone; clones share one in-process write lock.
///
/// Each entry is one `write` on an `O_APPEND` handle, so concurrent appends (also from another process) never
/// interleave within a line. The API is synchronous and touches a small file; call it from `spawn_blocking` if
/// that matters.
#[derive(Debug, Clone)]
pub struct Ledger {
    path: PathBuf,
    write_lock: Arc<Mutex<()>>,
}

impl Ledger {
    /// Opens (lazily) the ledger at `path`, e.g. `<data dir>/mcp-sfxgen/credits.jsonl`. Nothing is created until the
    /// first append.
    pub fn open(path: impl Into<PathBuf>) -> Self {
        Ledger {
            path: path.into(),
            write_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends one entry, creating parent directories as needed.
    pub fn append(&self, entry: &Entry) -> Result<(), LedgerError> {
        let row = Row {
            ts: entry.ts.to_rfc3339_opts(SecondsFormat::Millis, false),
            model: entry.model.clone(),
            credits: entry.credits,
            duration_s: entry.duration_s.filter(|d| d.is_finite()),
            file: entry.file.clone(),
            request_id: entry.request_id.clone(),
        };
        let mut line =
            serde_json::to_string(&row).map_err(|e| LedgerError::Encode(e.to_string()))?;
        line.push('\n');

        let io_err = |source| LedgerError::Io {
            path: self.path.clone(),
            source,
        };
        if let Some(parent) = self.path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(io_err)?;
        }
        let _guard = self.write_lock.lock().unwrap_or_else(|p| p.into_inner());
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(io_err)?;
        file.write_all(line.as_bytes()).map_err(io_err)?;
        Ok(())
    }

    /// Credits spent on the local calendar day of now.
    pub fn today_total(&self) -> Result<u64, LedgerError> {
        self.today_total_at(Local::now())
    }

    /// Credits spent in the local calendar month of now.
    pub fn month_total(&self) -> Result<u64, LedgerError> {
        self.month_total_at(Local::now())
    }

    /// Both totals from one read of the file.
    pub fn totals(&self) -> Result<Totals, LedgerError> {
        self.totals_at(Local::now())
    }

    /// [`Ledger::today_total`] as of `now` (for tests and for callers that already hold the time).
    pub fn today_total_at(&self, now: DateTime<Local>) -> Result<u64, LedgerError> {
        Ok(self.totals_at(now)?.today)
    }

    /// [`Ledger::month_total`] as of `now`.
    pub fn month_total_at(&self, now: DateTime<Local>) -> Result<u64, LedgerError> {
        Ok(self.totals_at(now)?.month)
    }

    pub fn totals_at(&self, now: DateTime<Local>) -> Result<Totals, LedgerError> {
        let text = match std::fs::read(&self.path) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(source) => {
                return Err(LedgerError::Io {
                    path: self.path.clone(),
                    source,
                });
            }
        };
        let today = now.date_naive();
        let mut totals = Totals { today: 0, month: 0 };
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let parsed = serde_json::from_str::<Row>(line).ok().and_then(|row| {
                DateTime::parse_from_rfc3339(&row.ts)
                    .ok()
                    .map(|ts| (ts.with_timezone(&Local), row.credits))
            });
            let Some((ts, credits)) = parsed else {
                tracing::warn!(
                    path = %self.path.display(),
                    line = index + 1,
                    "skipping corrupt credit ledger line"
                );
                continue;
            };
            let day = ts.date_naive();
            if day.year() == today.year() && day.month() == today.month() {
                totals.month = totals.month.saturating_add(u64::from(credits));
                if day == today {
                    totals.today = totals.today.saturating_add(u64::from(credits));
                }
            }
        }
        Ok(totals)
    }
}

/// Cost rates, from the `[elevenlabs]` config table (measured on James's Free plan, 2026-10-03).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CostRates {
    /// Credits per second of requested audio when `duration_seconds` is given.
    pub credits_per_second: u32,
    /// Credits per clip when the model picks the length.
    pub credits_auto_duration: u32,
}

impl Default for CostRates {
    fn default() -> Self {
        CostRates {
            credits_per_second: 10,
            credits_auto_duration: 50,
        }
    }
}

/// Credits a call will cost: `ceil(duration · credits_per_second) · variations`, or
/// `credits_auto_duration · variations` when no duration is given. A non-finite or non-positive duration is priced
/// as automatic (the caller should have rejected it). Saturates instead of overflowing.
pub fn estimate(duration_s: Option<f64>, variations: u32, rates: &CostRates) -> u32 {
    let per_clip = match duration_s.filter(|d| d.is_finite() && *d > 0.0) {
        Some(d) => {
            // Round away float noise first: 1.1 * 10 is 11.000000000000002, which must not price as 12.
            let raw = (d * f64::from(rates.credits_per_second) * 1e6).round() / 1e6;
            raw.ceil().min(f64::from(u32::MAX)) as u32
        }
        None => rates.credits_auto_duration,
    };
    per_clip.saturating_mul(variations)
}

/// Which limit a call would break.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    PerCall,
    Daily,
}

/// A call refused before anything was sent.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{}", self.message())]
pub struct LimitError {
    pub which: LimitKind,
    pub estimate: u32,
    pub limit: u32,
    pub today: u64,
}

impl LimitError {
    /// The tool-result error code (design §7).
    pub fn code(&self) -> &'static str {
        "credit_limit"
    }

    /// The config key that raises this limit.
    pub fn config_key(&self) -> &'static str {
        match self.which {
            LimitKind::PerCall => "per_call_credit_limit",
            LimitKind::Daily => "daily_credit_limit",
        }
    }

    fn message(&self) -> String {
        match self.which {
            LimitKind::PerCall => format!(
                "this call is estimated at {} credits, over the per-call limit of {}. Nothing was sent. \
                 Shorten the sound or ask for fewer variations, or raise `per_call_credit_limit` in the \
                 [elevenlabs] table of the sfxgen config.toml.",
                self.estimate, self.limit
            ),
            LimitKind::Daily => format!(
                "this call is estimated at {} credits and {} were already spent today, over the daily budget of {}. \
                 Nothing was sent. Wait until tomorrow, or raise `daily_credit_limit` in the [elevenlabs] table of \
                 the sfxgen config.toml.",
                self.estimate, self.today, self.limit
            ),
        }
    }

    /// How to fix it, as its own line for the tool result's `fix` field.
    pub fn fix(&self) -> String {
        format!(
            "Raise `{}` in the [elevenlabs] table of the sfxgen config.toml, or reduce the request.",
            self.config_key()
        )
    }
}

/// Refuses a call whose estimate is over the per-call ceiling, or that would take today's spend over the daily
/// budget. The per-call limit is checked first. A limit of 0 refuses everything.
pub fn check_limits(
    estimate: u32,
    per_call_limit: u32,
    daily_limit: u32,
    today_total: u64,
) -> Result<(), LimitError> {
    if estimate > per_call_limit {
        return Err(LimitError {
            which: LimitKind::PerCall,
            estimate,
            limit: per_call_limit,
            today: today_total,
        });
    }
    if today_total.saturating_add(u64::from(estimate)) > u64::from(daily_limit) {
        return Err(LimitError {
            which: LimitKind::Daily,
            estimate,
            limit: daily_limit,
            today: today_total,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn at(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Local> {
        Local
            .with_ymd_and_hms(y, mo, d, h, mi, 0)
            .earliest()
            .unwrap()
    }

    fn entry(ts: DateTime<Local>, credits: u32) -> Entry {
        Entry {
            ts,
            model: "elevenlabs-sfx-v2".into(),
            credits,
            duration_s: Some(1.0),
            file: Some("horn_01.wav".into()),
            request_id: Some("req123".into()),
        }
    }

    #[test]
    fn estimate_rules() {
        let rates = CostRates::default();
        assert_eq!(estimate(Some(1.0), 1, &rates), 10);
        assert_eq!(estimate(Some(2.5), 1, &rates), 25);
        assert_eq!(estimate(Some(0.5), 1, &rates), 5);
        assert_eq!(estimate(Some(7.5), 4, &rates), 300);
        assert_eq!(
            estimate(Some(1.1), 1, &rates),
            11,
            "float noise must not round up"
        );
        assert_eq!(
            estimate(Some(0.55), 1, &rates),
            6,
            "a real fraction rounds up"
        );
        assert_eq!(estimate(None, 1, &rates), 50);
        assert_eq!(estimate(None, 3, &rates), 150);
        assert_eq!(estimate(Some(f64::NAN), 2, &rates), 100);
        assert_eq!(estimate(Some(-1.0), 1, &rates), 50);
        let custom = CostRates {
            credits_per_second: 40,
            credits_auto_duration: 200,
        };
        assert_eq!(estimate(Some(1.0), 2, &custom), 80);
        assert_eq!(estimate(Some(1e12), 4, &rates), u32::MAX);
        assert_eq!(estimate(Some(1.0), 0, &rates), 0);
    }

    #[test]
    fn limits() {
        assert!(check_limits(300, 300, 2000, 0).is_ok());
        assert!(check_limits(300, 300, 2000, 1700).is_ok());

        let e = check_limits(301, 300, 2000, 0).unwrap_err();
        assert_eq!(e.which, LimitKind::PerCall);
        assert_eq!((e.estimate, e.limit, e.today), (301, 300, 0));
        let msg = e.to_string();
        assert!(
            msg.contains("per_call_credit_limit") && msg.contains("[elevenlabs]"),
            "{msg}"
        );
        assert_eq!(e.code(), "credit_limit");

        let e = check_limits(300, 300, 2000, 1701).unwrap_err();
        assert_eq!(e.which, LimitKind::Daily);
        assert_eq!((e.estimate, e.limit, e.today), (300, 2000, 1701));
        let msg = e.to_string();
        assert!(
            msg.contains("daily_credit_limit") && msg.contains("1701"),
            "{msg}"
        );
        assert!(e.fix().contains("daily_credit_limit"));

        // Per-call is reported first when both are broken.
        assert_eq!(
            check_limits(500, 300, 100, 0).unwrap_err().which,
            LimitKind::PerCall
        );
        assert!(
            check_limits(1, 0, 0, 0).is_err(),
            "a zero limit refuses everything"
        );
    }

    #[test]
    fn totals_split_by_local_day_and_month() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(dir.path().join("nested/mcp-sfxgen/credits.jsonl"));
        assert_eq!(
            ledger.today_total().unwrap(),
            0,
            "missing file is an empty ledger"
        );

        ledger.append(&entry(at(2026, 9, 29, 12, 0), 1000)).unwrap(); // earlier month
        ledger.append(&entry(at(2026, 9, 30, 23, 59), 7)).unwrap(); // last minute of September
        ledger.append(&entry(at(2026, 10, 1, 0, 1), 11)).unwrap(); // first minute of October
        ledger.append(&entry(at(2026, 10, 2, 23, 30), 20)).unwrap();
        ledger.append(&entry(at(2026, 10, 3, 0, 30), 30)).unwrap(); // just after midnight
        ledger.append(&entry(at(2026, 10, 3, 9, 0), 40)).unwrap();
        ledger.append(&entry(at(2025, 10, 3, 9, 0), 5000)).unwrap(); // same month, last year

        let now = at(2026, 10, 3, 18, 0);
        assert_eq!(ledger.today_total_at(now).unwrap(), 70);
        assert_eq!(ledger.month_total_at(now).unwrap(), 11 + 20 + 30 + 40);
        assert_eq!(
            ledger.totals_at(now).unwrap(),
            Totals {
                today: 70,
                month: 101
            }
        );

        // The next minute after midnight, the day resets but the month doesn't.
        let next_day = at(2026, 10, 4, 0, 5);
        assert_eq!(ledger.today_total_at(next_day).unwrap(), 0);
        assert_eq!(ledger.month_total_at(next_day).unwrap(), 101);

        // And on the first of November everything resets.
        let november = at(2026, 11, 1, 0, 0);
        assert_eq!(ledger.month_total_at(november).unwrap(), 0);
        // But the day before, September's last minute counts for September only.
        assert_eq!(
            ledger.month_total_at(at(2026, 9, 30, 23, 59)).unwrap(),
            1007
        );
    }

    #[test]
    fn entries_use_offsets_of_other_zones() {
        // A UTC stamp is bucketed by the *local* date of that instant.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("credits.jsonl");
        let instant = at(2026, 10, 3, 12, 0);
        let utc = instant.with_timezone(&chrono::Utc).to_rfc3339();
        std::fs::write(
            &path,
            format!("{{\"ts\":\"{utc}\",\"model\":\"m\",\"credits\":9}}\n"),
        )
        .unwrap();
        assert_eq!(Ledger::open(&path).today_total_at(instant).unwrap(), 9);
    }

    #[test]
    fn corrupt_lines_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("credits.jsonl");
        let ledger = Ledger::open(&path);
        let now = at(2026, 10, 3, 12, 0);
        ledger.append(&entry(now, 10)).unwrap();
        // Garbage, a truncated line, a wrong type, a bad timestamp, blank lines and invalid UTF-8.
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"not json\n{\"ts\":\"2026-10-03T12:00:00+03:00\",\"model\":\"m\",\"cre\n")
            .unwrap();
        f.write_all(
            b"{\"ts\":\"2026-10-03T12:00:00+03:00\",\"model\":\"m\",\"credits\":\"ten\"}\n",
        )
        .unwrap();
        f.write_all(b"{\"ts\":\"yesterday\",\"model\":\"m\",\"credits\":99}\n\n   \n")
            .unwrap();
        f.write_all(b"\xff\xfe\n").unwrap();
        drop(f);
        ledger.append(&entry(now, 5)).unwrap();
        assert_eq!(ledger.today_total_at(now).unwrap(), 15);
    }

    #[test]
    fn rows_round_trip_and_stay_one_line() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(dir.path().join("credits.jsonl"));
        let mut e = entry(at(2026, 10, 3, 12, 0), 10);
        e.file = Some("line\nbreak \"quoted\".wav".into());
        ledger.append(&e).unwrap();
        ledger.append(&Entry::now("m", 3)).unwrap();
        let text = std::fs::read_to_string(ledger.path()).unwrap();
        assert_eq!(text.lines().count(), 2);
        let first: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(first["credits"], 10);
        assert_eq!(first["file"], "line\nbreak \"quoted\".wav");
        assert_eq!(first["request_id"], "req123");
        assert!(
            first["ts"]
                .as_str()
                .unwrap()
                .starts_with("2026-10-03T12:00:00.000")
        );
        let second: serde_json::Value = serde_json::from_str(text.lines().nth(1).unwrap()).unwrap();
        assert!(second.get("file").is_none() && second.get("duration_s").is_none());
    }

    #[test]
    fn concurrent_appends_do_not_interleave() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(dir.path().join("credits.jsonl"));
        let now = Local::now();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let ledger = ledger.clone();
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        ledger.append(&Entry::now("m", 1)).unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            ledger.today_total_at(now).unwrap(),
            400,
            "no line was lost or torn"
        );
    }
}
