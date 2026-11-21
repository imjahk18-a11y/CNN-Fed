import hashlib
import json

import numpy as np
import pytest
import torch

from experiments.jetson_virtual_bench import budget_check
from federated.edge_batches import EdgeBatches


def test_client_shards_preserve_order_and_detect_changes(tmp_path):
    entries = []
    for i in range(2):
        p = tmp_path / f"{i}.npz"
        np.savez(p, coords=np.array([[0,0,33],[1,0,33]],np.int32),
                 rgb=np.array([[0,127,255],[255,0,0]],np.float32),
                 labels=np.array([i,5],np.uint8))
        entries.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(dict(role="client1",patches=entries)))
    data = EdgeBatches(m,"client1",2,"cpu")
    g,x,y = data[0]
    assert g.coords.tolist()==[[0,0,0,33],[0,1,0,33],[1,0,0,33],[1,1,0,33]]
    assert torch.allclose(x[0],torch.tensor([0,127/255,1]))
    assert y.tolist()==[0,5,1,5]
    with pytest.raises(ValueError,match="Role"):
        EdgeBatches(m,"client2",2,"cpu")
    (tmp_path/"0.npz").write_bytes(b"damaged")
    with pytest.raises(ValueError,match="Corrupt"):
        data[0]


def test_budget_accounts_for_driver_and_host():
    sample=dict(driver_sample_mib=500, current_reserved_mib=100, peak_reserved_mib=200, peak_rss_mib=2500)
    assert budget_check(sample,4,1,2)==dict(working_set_proxy_mib=3100,budget_status="EXCEEDS_BUDGET")
    assert budget_check(sample,8,1,6)["budget_status"]=="PASS"
    assert budget_check(dict(sample,driver_sample_mib=None),4,1,2)["budget_status"]=="INCOMPLETE"
