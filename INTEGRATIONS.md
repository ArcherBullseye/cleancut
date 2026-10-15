# Local Radarr / Sonarr / SABnzbd workflow (beta.15)

Implemented locally; **disabled by default**. Deploy and configure only after
the current AI Mac queue is finished. Opening Integrations does not install
notifications, change Arr settings, or touch media.

## Workflow

1. Add the `cleancut` tag to a movie in Radarr or a series in Sonarr.
2. SABnzbd downloads/unpacks normally. Radarr/Sonarr imports the library file.
3. An authenticated per-file import/upgrade webhook submits the file. CleanCut
   independently checks its ID, owner and tag against the saved Arr service;
   webhook-supplied filesystem paths are not trusted.
4. CleanCut maps the canonical library path to the Mac mount and scans locally.
   By default it waits for you to review the scan and click **Render**, approving
   library replacement. Automatic rendering is a separate integration option.
5. The render stages locally in the original container. Full decode, edited
   duration, resolution and audio-presence checks are mandatory.
6. Source identity/content and Arr ownership are checked again. CleanCut uploads
   a checksum-verified hidden sibling and makes a verified temporary NAS backup.
   One same-directory rename replaces the original. A confirmed Arr rescan is
   required before the backup is deleted.

Ordinary Library jobs still create separate `.clean.mp4` copies. Only integration
jobs replace library originals; their review page shows a warning. Scan/render
settings are snapshotted at intake; replacement verification cannot be disabled.
No additional AI model is required.

## Setup — after existing jobs finish

Open **Integrations** on the CleanCut Mac using `http://localhost:3000`.
Administration is loopback-only: use VNC or an SSH tunnel. Webhooks accept LAN
traffic with their own password.

1. Enter each Arr URL and API key. Use literal private/loopback IPs; hostnames,
   external addresses, redirects and environment proxies are rejected. URL
   bases such as `/radarr` are supported.
2. Set server and local roots; test a complete imported file path. Mount the NAS
   in macOS first. Missing media directories are not created. Symlinks/escapes
   are rejected. Supported replacement containers: MKV, MP4, M4V, MOV.
3. Save and test each service, then enable imports. Leave automatic rendering
   off for the first end-to-end test.
4. When the queue is idle, start a LAN listener:

   ```sh
   CLEANCUT_HOST=0.0.0.0 ./macos/run.sh
   ```

   Do not port-forward. The existing job/browser UI assumes a trusted LAN;
   webhook authentication is not general application login. Use a firewall,
   tunnel or authenticated HTTPS proxy on untrusted networks. Plain HTTP sends
   credentials without transport encryption.
5. In each Arr app: **Settings → Connect → Webhook**, method POST:

   ```text
   http://AI-MAC-IP:3000/api/integrations/radarr/webhook
   http://AI-MAC-IP:3000/api/integrations/sonarr/webhook
   ```

   Username: `cleancut`. Password: **Show webhook password** in Integrations.
   Set the notification's tag filter to the same saved opt-in tag.
   Enable per-file import/download and upgrade triggers. Disable Grab, Rename,
   and Sonarr's aggregate Import Complete. Test and save the notification.
   Exact trigger labels vary between Arr versions.
6. Apply the tag when adding movies/series. Sonarr's tag is series-wide, not
   per-episode. This cleans your import; it does not find pre-censored releases.
   Adding a tag to an already-imported title does not retroactively queue it.

Keys live in `DATA_DIR/integrations.json`, created with mode 0600. Blank key
fields retain saved values. Normal settings responses and jobs never include
these credentials. Rotate the webhook password only when you can update both
Arr notifications.

### This installation's discovered mapping

Reference values only; these are not automatically enabled:

| Service | Address / path |
| --- | --- |
| AI Mac | `192.168.88.157` |
| Radarr | `http://192.168.88.237:7878` |
| Sonarr | `http://192.168.88.237:8989` |
| NAS | `192.168.88.89` |
| Server → Mac root | `/nas` → `/Volumes/Public` |
| Movies | `/nas/movies` → `/Volumes/Public/movies` |
| Shows | `/nas/shows` → `/Volumes/Public/shows` |

Both Mac folders reported mounted/writable during read-only setup checks.
Recheck after reboot; macOS may assign another mount name. The separate
`/Volumes/Media` mount is not this mapping.

## SABnzbd

Keep the existing Arr-managed `movies` and `shows` categories. No new script,
model, category or public SAB endpoint is needed. Trigger only after Arr's
library import, not also after SAB completion. Standalone SAB downloads that
bypass Arr are not supported by these webhooks; use the ordinary folder queue.

## Failure and recovery

The SQLite `imports` journal records identity, SHA-256 hashes, job IDs, recovery
paths and rescan IDs. It survives restart and job-history deletion. Repeated
deliveries reuse the same job; already-published output is recognized to avoid
rescan loops. An upgrade with a new fingerprint queues behind the old version;
the older job's checks refuse replacement of the changed source.

- Scan/render failure before publication: original unchanged. Use **Retry scan**
  for a retained failed/canceled scan, or render an existing completed scan again.
  Duplicate webhook delivery does not silently retry failed work.
- NAS full/disconnected or crash during copies: local render and journal remain.
  Restore the mount and use **Recover**.
- Crash around rename: recovery compares hashes. If output is already in place,
  it confirms/retries the rescan. If the original is intact, it removes this
  attempt's temporary copies (including partial copies) and permits re-render.
- Rescan failure: cleaned file remains in the library; original backup stays.
  Recovery never blindly rolls back over a potentially newer import. Recovery
  is explicit, not an automatic startup operation.
- Missing source or a file matching neither version: no automatic overwrite or
  deletion. Inspect the mount, library and backup manually before restoring.
- Unresolved publication jobs cannot be canceled/deleted, including by bulk
  cleanup. Completed jobs can be deleted; their deduplication journal remains.

Temporary NAS files: `.ORIGINALNAME.cleancut-ID.backup` and
`.STEM.cleancut-ID.upload.EXT`. Normal success removes both. Keep backups until
recovery confirms the current file; do not blindly rename them over newer media.

## Operational limits

- Budget disk space for the local render and both a backup and upload on the NAS.
  Full decoding and hashing intentionally add I/O.
- Do not rename/move/upgrade the same file during publication. Identity, API and
  content checks run immediately before rename, but NAS/Arr has no shared
  compare-and-swap lock. An external writer in that final check/rename window
  cannot be completely excluded. A backup is recovery protection, not a
  distributed transaction guarantee.
- Filename/container are retained; Arr release-name/quality labels may still
  describe the original. The pipeline processes selected audio/subtitles, not
  every alternate track, chapter or attachment. External subtitle sidecars may
  no longer align after scene cuts.
- Successful rendering does not prove every objectionable scene was detected.
  Review weak visual detections. Experimental background preservation may
  still leak faint speech.

## Development checks

```sh
.venv/bin/python -m pytest -q
node --test tests/test_library_ui.cjs tests/test_integrations_ui.cjs
```

Fixtures use temporary directories and mocked Arr APIs, never the production
NAS/AI Mac. Real MKV/MP4 replacement tests additionally run when local FFmpeg and
ffprobe are available.

## Primary protocol references

- [Radarr webhook implementation](https://github.com/Radarr/Radarr/blob/develop/src/NzbDrone.Core/Notifications/Webhook/Webhook.cs)
- [Radarr import payload](https://github.com/Radarr/Radarr/blob/develop/src/NzbDrone.Core/Notifications/Webhook/WebhookImportPayload.cs)
- [Webhook authentication settings](https://github.com/Radarr/Radarr/blob/develop/src/NzbDrone.Core/Notifications/Webhook/WebhookSettings.cs)
- [Sonarr webhook implementation](https://github.com/Sonarr/Sonarr/blob/develop/src/NzbDrone.Core/Notifications/Webhook/Webhook.cs)
- [Sonarr import payload](https://github.com/Sonarr/Sonarr/blob/develop/src/NzbDrone.Core/Notifications/Webhook/WebhookImportPayload.cs)
- [SABnzbd categories](https://sabnzbd.org/wiki/configuration/5.1/categories)
