# Immich + Immich Power Tools

This folder keeps the local Immich deployment together:

- `docker-compose.yml` runs Immich, Postgres, Valkey, machine learning, and Immich Power Tools.
- `.env` is the real local runtime config and is intentionally gitignored.
- `.env.restore` is the temporary bootstrap config used only for restoring the database dump and is intentionally gitignored.
- `restore.ps1` restores the provided `pg_dumpall`-style backup into a fresh Postgres cluster, then switches the stack back to the normal runtime credentials.

## First-time setup

1. Copy `.env.example` to `.env` and set:
   - `PHOTO_ALBUMS_LOCATION`
   - `DB_PASSWORD`
   - `EXTERNAL_IMMICH_URL`
   - `IMMICH_API_KEY`
   - `JWT_SECRET`
2. Copy `.env.restore.example` to `.env.restore` and set a temporary `DB_PASSWORD` different from the runtime password.
3. Run the restore:

   ```powershell
   .\restore.ps1
   ```

4. Open:
   - Immich: `http://<host>:2283`
   - Immich Power Tools: `http://<host>:3000`

## Why the restore script is shaped this way

The backup at `~/immich-db-backup-5-8-2026-1138.sql` is a cluster dump, not a single-database dump. It contains statements such as `DROP DATABASE immich;`, `DROP ROLE postgres;`, and recreation of `template1`. Restoring that directly into a running Immich database as `postgres` is the wrong starting state: the role and database already exist, and the active login cannot drop itself.

`restore.ps1` instead starts a clean cluster under a temporary superuser, creates the placeholder `postgres` role and `immich` database that the dump expects to drop, restores the full dump, resets the restored `postgres` password to the value from `.env`, then starts the normal stack.

## Remote access

The Compose file publishes Immich on port `2283` and Immich Power Tools on port `3000` on all host interfaces. Power Tools talks to Immich over Docker's internal network via `IMMICH_URL=http://immich-server:2283` and `DB_HOST=database`; browser-facing links and redirects use `EXTERNAL_IMMICH_URL`.

## External photo library

`PHOTO_ALBUMS_LOCATION` is mounted read-only into the Immich server at `/cordell_photos`. In Immich, add `/cordell_photos` as an external library path when you want Immich to index the raw Photo Albums tree without taking ownership of it.
