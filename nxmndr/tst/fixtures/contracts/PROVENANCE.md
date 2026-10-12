# Contract fixtures copied from the plugin

The backend repository cannot import the plugin's `contracts` package, so contract
fixtures are copied here verbatim. Each copy is identified by its path in the plugin
repository and its SHA-256, which `tst/unit/test_stream_context_fixture.py` pins; the
hash identifies the plugin's file in whatever commit it is found. When the plugin
changes a fixture, copy it again and update this table and the pinned hash together.

| File | Source in the plugin repository (`nxmndr-qgis`) | SHA-256 |
|---|---|---|
| `stream_context_v1.json` | `src/contracts/fixtures/stream_context_v1.json` | `212eeb8207afc688c012ba56733824b1842b556c26c698e2841e8fa8f1466e06` |
