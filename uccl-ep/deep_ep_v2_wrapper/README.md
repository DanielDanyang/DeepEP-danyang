# DeepEP V2 Wrapper for AWS EFA

This wrapper follows the same packaging idea as UCCL-EP's original
`deep_ep_wrapper`: it installs a `deep_ep` Python package that exposes a
DeepEP-compatible API while routing AWS EFA communication through the UCCL-style
proxy backend.

This wrapper targets the DeepEP V2 `ElasticBuffer` API.

## Intended install flow

```bash
cd uccl-ep
python setup.py install

cd deep_ep_v2_wrapper
python setup.py install
```

The first command installs the native `uccl.ep` extension. The second command
installs a `deep_ep` package that exposes `ElasticBuffer` and V2 handle objects.

## Porting notes

The initial implementation is intentionally narrow:

- AWS EFA only.
- EP16 / p5en first.
- `ElasticBuffer.dispatch` and `ElasticBuffer.combine` first.
- Other DeepEP V2 APIs should fail explicitly until implemented.

