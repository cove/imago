# FamilySearch export

Downloads the Cove Schneider FamilySearch ancestor tree as a GEDCOM file with `getmyancestors`.

Run:

```sh
just familysearch-download
```

The default start person is `P631-4WH`, and the default generation cap is `200`. `getmyancestors` stops early when it runs out of parents, so this cap is intended to mean "as far back as FamilySearch can provide for this line."

The password is read from `FAMILYSEARCH_PASSWORD`, which is managed through the encrypted Nushell secrets template in chezmoi.
