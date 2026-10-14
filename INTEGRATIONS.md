# Radarr / Sonarr / SABnzbd integration — proposed, not implemented

The requested policy is to **replace the library copy only after successful
cleaning**, rather than keeping two permanent versions. No source-replacement
code or webhook endpoint is enabled by beta.14.

## Proposed user workflow

1. Add a movie in Radarr or a series in Sonarr with a `cleancut` tag to opt in.
   Sonarr's series tag applies to the show's imports; this is not a native
   per-episode “clean release” selector and does not find pre-edited downloads.
2. SABnzbd downloads/unpacks normally; Radarr/Sonarr imports the final library file.
3. A tag-filtered import/upgrade webhook submits that imported file to CleanCut.
   Trigger after import, not both SAB completion and import: the download file
   may move, and duplicate triggers would queue duplicate cleaning.
4. CleanCut maps the server path to the mounted NAS path on the AI Mac, scans
   and renders locally into staging. The imported source stays untouched on failure.
5. After validating the complete output and confirming the source has not changed
   (e.g. an intervening upgrade), publish with rollback protection and rescan the
   movie/series in the owning app. A temporary backup is needed until this succeeds;
   it is not a second permanent library edition.

## Requirements before implementing replacement

- Local URLs and server-to-Mac NAS path mappings, with a connection/path test.
  Credentials should be entered in protected settings, not chat or logs.
- Authenticated inbound webhooks, explicit allowed local services, and safe media
  root checks. No arbitrary filesystem paths or URLs supplied by a webhook.
- Durable event deduplication by app, file identity and source fingerprint.
- Full decode validation, timeline/stream checks, adequate disk space and writable
  NAS destination; processing success does not guarantee every objectionable
  scene was detected. Weak visual detections still require review.
- Preserve the correct container/extension, verify file identity immediately
  before replacement, and coordinate rescans/upgrades. Never just rename MP4
  bytes to `.mkv` or overwrite a newer import with an older cleaned result.
- Recoverable publication on NAS disconnect, cancellation, process crash or
  failed library rescan; restart recovery and stale-job handling.
- Tests using captured Test/Import/Upgrade payloads, repeated deliveries, file
  moves/upgrades, mapping escapes, failed render, failed copy and recovery.

SABnzbd also supports category-specific post-processing scripts. Those would
be a separate optional path for downloads *not* managed by Radarr/Sonarr, not
an additional hook for the same managed download.

## Primary references inspected

- [Radarr webhook events](https://github.com/Radarr/Radarr/blob/develop/src/NzbDrone.Core/Notifications/Webhook/Webhook.cs)
- [Radarr notification tag filtering](https://github.com/Radarr/Radarr/blob/develop/src/NzbDrone.Core/Notifications/NotificationService.cs)
- [Sonarr webhook events](https://github.com/Sonarr/Sonarr/blob/develop/src/NzbDrone.Core/Notifications/Webhook/Webhook.cs)
- [Sonarr notification tag filtering](https://github.com/Sonarr/Sonarr/blob/develop/src/NzbDrone.Core/Notifications/NotificationService.cs)
- [SABnzbd categories](https://sabnzbd.org/wiki/configuration/5.1/categories)
