"""达标结果交接（handoff）端到端集成测试：需要真实 Redis（REDIS_URL）。

租约相关用例依赖一个很短的租约窗口：
    REDIS_URL=redis://localhost:6399/15 \
    HANDOFF_LEASE_TTL_MS=600 \
    BARRIER_POLL_INTERVAL_S=0.05 python3 -m pytest -s tests/test_handoffs.py
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

# 在导入 app 前固定配置：短租约（0.6s）便于到期接管测试
os.environ.setdefault("REDIS_URL", "redis://localhost:6399/15")
os.environ.setdefault("BARRIER_POLL_INTERVAL_S", "0.05")
os.environ.setdefault("HANDOFF_LEASE_TTL_MS", "600")

from app.main import app  # noqa: E402

API = "http://test"


@pytest.fixture(autouse=True)
async def _lifespan():
    async with LifespanManager(app):
        yield


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url=API)


async def _reached_barrier(client, participants=("a", "b"), threshold=None, ttl_s=30):
    """创建屏障并让其达标，返回 (bid, quorum, frozen_at)。"""
    deadline = (
        datetime.now(timezone.utc) + timedelta(seconds=ttl_s)
    ).isoformat()
    r = await client.post(
        "/barriers",
        json={
            "participants": list(participants),
            "threshold": threshold if threshold is not None else len(participants),
            "deadline": deadline,
        },
    )
    assert r.status_code == 201, r.text
    bid = r.json()["barrier_id"]
    for p in participants:
        await client.post(f"/barriers/{bid}/checkin", json={"participant_id": p})
    view = (await client.get(f"/barriers/{bid}")).json()
    assert view["status"] == "reached"
    return bid, view["quorum"], view["frozen_at"]


async def create_handoff(client, bid: str):
    return await client.post("/handoffs", json={"barrier_id": bid})


async def claim(client, hid: str, owner: str):
    return await client.post(f"/handoffs/{hid}/claim", json={"owner": owner})


async def renew(client, hid: str, owner: str, fence: int):
    return await client.post(
        f"/handoffs/{hid}/renew", json={"owner": owner, "fence": fence}
    )


async def complete(client, hid: str, owner: str, fence: int):
    return await client.post(
        f"/handoffs/{hid}/complete", json={"owner": owner, "fence": fence}
    )


# ---------------------------------------------------------------------------
# 创建
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_handoff_copies_quorum_and_frozen_at():
    """为已达标屏障建单：201，复制 quorum/frozen_at，初始 idle、无租约。"""
    async with await _client() as c:
        bid, quorum, frozen_at = await _reached_barrier(c, ["a", "b", "c"])
        r = await create_handoff(c, bid)
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["replayed"] is False
        assert body["status"] == "idle"
        assert body["quorum"] == quorum == ["a", "b", "c"]
        assert body["frozen_at"] == frozen_at
        assert body["barrier_id"] == bid
        assert body["owner"] is None
        assert body["fence"] is None
        assert body["lease_expires_at"] is None
        assert body["lease_active"] is False
        assert body["completed_at"] is None
        assert body["handoff_id"]


@pytest.mark.asyncio
async def test_create_handoff_unknown_barrier_404():
    async with await _client() as c:
        r = await create_handoff(c, "deadbeefdeadbeef")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_create_handoff_not_reached_409():
    """waiting / cancelled / expired 三种未达标情况都返回 409。"""
    async with await _client() as c:
        # waiting
        deadline = (
            datetime.now(timezone.utc) + timedelta(seconds=30)
        ).isoformat()
        r = await c.post(
            "/barriers",
            json={"participants": ["a", "b"], "deadline": deadline},
        )
        bid_waiting = r.json()["barrier_id"]
        assert (await create_handoff(c, bid_waiting)).status_code == 409

        # cancelled
        r = await c.post(
            "/barriers",
            json={"participants": ["x"], "deadline": deadline},
        )
        bid_cancelled = r.json()["barrier_id"]
        assert (
            await c.post(
                f"/barriers/{bid_cancelled}/cancel", json={"reason": "停"}
            )
        ).status_code == 200
        assert (await create_handoff(c, bid_cancelled)).status_code == 409

        # expired
        r = await c.post(
            "/barriers",
            json={
                "participants": ["y"],
                "deadline": (
                    datetime.now(timezone.utc) + timedelta(seconds=1)
                ).isoformat(),
            },
        )
        bid_expired = r.json()["barrier_id"]
        await asyncio.sleep(1.2)
        assert (await c.get(f"/barriers/{bid_expired}")).json()["status"] == "expired"
        assert (await create_handoff(c, bid_expired)).status_code == 409


@pytest.mark.asyncio
async def test_create_handoff_idempotent_same_handoff():
    """同一屏障重复创建：200 + replayed=true，返回同一份交接单。"""
    async with await _client() as c:
        bid, quorum, frozen_at = await _reached_barrier(c)
        r1 = await create_handoff(c, bid)
        assert r1.status_code == 201
        first = r1.json()

        r2 = await create_handoff(c, bid)
        assert r2.status_code == 200, r2.text
        second = r2.json()
        assert second["replayed"] is True
        assert second["handoff_id"] == first["handoff_id"]
        assert second["quorum"] == quorum
        assert second["frozen_at"] == frozen_at
        # 无论重复多少次都是同一单
        r3 = await create_handoff(c, bid)
        assert r3.status_code == 200
        assert r3.json()["handoff_id"] == first["handoff_id"]


@pytest.mark.asyncio
async def test_concurrent_create_only_one_handoff():
    """同一屏障并发创建：Redis 原子去重，全部返回同一个 handoff_id。"""
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c, ["p", "q"])
        results = await asyncio.gather(*(create_handoff(c, bid) for _ in range(8)))
        assert all(r.status_code in (200, 201) for r in results), results
        ids = {r.json()["handoff_id"] for r in results}
        assert len(ids) == 1, ids
        firsts = [r for r in results if r.status_code == 201]
        assert len(firsts) == 1, [r.status_code for r in results]


@pytest.mark.asyncio
async def test_two_barriers_get_distinct_handoffs():
    async with await _client() as c:
        bid1, _, _ = await _reached_barrier(c, ["a"])
        bid2, _, _ = await _reached_barrier(c, ["b"])
        h1 = (await create_handoff(c, bid1)).json()
        h2 = (await create_handoff(c, bid2)).json()
        assert h1["handoff_id"] != h2["handoff_id"]
        assert h1["quorum"] == ["a"] and h2["quorum"] == ["b"]


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_handoff_404():
    async with await _client() as c:
        assert (await c.get("/handoffs/deadbeefdeadbeef")).status_code == 404


@pytest.mark.asyncio
async def test_get_handoff_reflects_lease_lifecycle():
    """GET 在 idle / active / expired / completed 各阶段返回正确状态。"""
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]

        g = (await c.get(f"/handoffs/{hid}")).json()
        assert g["status"] == "idle" and g["lease_active"] is False

        r = await claim(c, hid, "worker-1")
        assert r.status_code == 200
        claimed = r.json()
        assert claimed["status"] == "active"
        assert claimed["lease_active"] is True
        assert claimed["owner"] == "worker-1"
        assert claimed["fence"] == 1

        g = (await c.get(f"/handoffs/{hid}")).json()
        assert g["status"] == "active" and g["lease_active"] is True
        assert g["fence"] == 1 and g["owner"] == "worker-1"

        await asyncio.sleep(0.8)  # 超过 0.6s 租约
        g = (await c.get(f"/handoffs/{hid}")).json()
        assert g["status"] == "expired"
        assert g["lease_active"] is False
        # 上一任持有者信息保留，但凭证已失效
        assert g["owner"] == "worker-1" and g["fence"] == 1


# ---------------------------------------------------------------------------
# claim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_returns_monotonic_fence_and_expiry():
    """认领返回单调递增 fence；租约到期时间约为认领时刻 + 租约窗口。"""
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]

        t0 = time.time()
        r = await claim(c, hid, "w1")
        assert r.status_code == 200
        body = r.json()
        assert body["fence"] == 1
        expires = datetime.fromisoformat(body["lease_expires_at"])
        delta = (expires - datetime.fromtimestamp(t0, tz=timezone.utc)).total_seconds()
        assert 0.5 < delta < 0.9, delta  # 配置租约 600ms + 裁决耗时余量

        # 租约有效：同一 owner 重复认领也被拒绝
        assert (await claim(c, hid, "w1")).status_code == 409
        # 其他 owner 竞争同样被拒绝
        assert (await claim(c, hid, "w2")).status_code == 409


@pytest.mark.asyncio
async def test_concurrent_claim_single_winner():
    """空闲单据上并发认领：只有一个赢家拿到 fence=1，其余全部 409。"""
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]

        results = await asyncio.gather(
            *(claim(c, hid, f"w{i}") for i in range(10))
        )
        oks = [r for r in results if r.status_code == 200]
        conflicts = [r for r in results if r.status_code == 409]
        assert len(oks) == 1 and len(conflicts) == 9, [
            r.status_code for r in results
        ]
        assert oks[0].json()["fence"] == 1


@pytest.mark.asyncio
async def test_takeover_after_expiry_bumps_fence_and_revokes_old_credential():
    """到期接管：新 owner 认领成功，fence 递增；旧 owner 的旧凭证续租/完成 409。"""
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]

        first = (await claim(c, hid, "old")).json()
        assert first["fence"] == 1

        await asyncio.sleep(0.8)  # 租约到期

        # 旧凭证在到期后立即失效：续租 / 完成均 409
        assert (await renew(c, hid, "old", 1)).status_code == 409
        assert (await complete(c, hid, "old", 1)).status_code == 409

        # 新 owner 接管
        taken = await claim(c, hid, "new")
        assert taken.status_code == 200
        tbody = taken.json()
        assert tbody["owner"] == "new"
        assert tbody["fence"] == 2  # 单调递增
        assert tbody["status"] == "active"

        g = (await c.get(f"/handoffs/{hid}")).json()
        assert g["fence"] == 2 and g["owner"] == "new" and g["lease_active"] is True

        # 旧 owner 用旧 fence 一律被拒；新 owner 用新 fence 可续租
        assert (await renew(c, hid, "old", 1)).status_code == 409
        assert (await renew(c, hid, "new", 1)).status_code == 409  # 旧 fence 值
        assert (await renew(c, hid, "new", 2)).status_code == 200


@pytest.mark.asyncio
async def test_concurrent_takeover_after_expiry_single_winner_fence_bumps_once():
    """到期瞬间多个接管者并发：只有一个接管成功，fence 只递增一次。"""
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]
        assert (await claim(c, hid, "incumbent")).status_code == 200
        await asyncio.sleep(0.8)

        results = await asyncio.gather(
            *(claim(c, hid, f"c{i}") for i in range(8))
        )
        oks = [r for r in results if r.status_code == 200]
        assert len(oks) == 1, [r.status_code for r in results]
        body = oks[0].json()
        assert body["fence"] == 2
        assert all(r.status_code == 409 for r in results if r is not oks[0])


# ---------------------------------------------------------------------------
# renew
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_renew_extends_lease_with_valid_credential():
    """有效 owner+fence 续租成功，到期点向后顺延。"""
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]
        claimed = (await claim(c, hid, "w")).json()
        first_expires = claimed["lease_expires_at"]

        await asyncio.sleep(0.2)
        r = await renew(c, hid, "w", claimed["fence"])
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "active" and body["lease_active"] is True
        assert body["fence"] == 1  # 续租不推进 fence
        assert body["lease_expires_at"] > first_expires

        # 超过原租约窗口但在续租后的窗口内：仍应 active（证明确实顺延）
        await asyncio.sleep(0.5)
        g = (await c.get(f"/handoffs/{hid}")).json()
        assert g["status"] == "active" and g["lease_active"] is True


@pytest.mark.asyncio
async def test_renew_requires_owner_fence_and_active_lease():
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]

        # 未认领直接续租 409
        assert (await renew(c, hid, "w", 1)).status_code == 409

        assert (await claim(c, hid, "w")).status_code == 200
        assert (await renew(c, hid, "other", 1)).status_code == 409  # owner 不符
        assert (await renew(c, hid, "w", 2)).status_code == 409  # fence 错误
        assert (await renew(c, hid, "w", 0)).status_code == 422  # fence 非法
        assert (await renew(c, hid, "  ", 1)).status_code == 422  # owner 空白

        # 到期后续租 409
        await asyncio.sleep(0.8)
        assert (await renew(c, hid, "w", 1)).status_code == 409


# ---------------------------------------------------------------------------
# complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_terminal_blocks_everything():
    """完成后终态：completed_at 落盘，永久拒绝认领 / 续租 / 重复完成。"""
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]
        claimed = (await claim(c, hid, "w")).json()

        # 凭证错误不能完成
        assert (await complete(c, hid, "other", claimed["fence"])).status_code == 409
        assert (await complete(c, hid, "w", claimed["fence"] + 1)).status_code == 409

        r = await complete(c, hid, "w", claimed["fence"])
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        assert body["completed_at"] is not None
        assert body["lease_active"] is False
        completed_at = body["completed_at"]
        assert datetime.fromisoformat(completed_at).tzinfo is not None

        g = (await c.get(f"/handoffs/{hid}")).json()
        assert g["status"] == "completed"
        assert g["completed_at"] == completed_at

        # 终态永久拒绝：即使立即（租约原本仍有效）也不能续租 / 重复完成
        assert (await renew(c, hid, "w", claimed["fence"])).status_code == 409
        assert (await complete(c, hid, "w", claimed["fence"])).status_code == 409
        # 永久拒绝再次认领：即使等租约到期后仍然 409
        assert (await claim(c, hid, "new")).status_code == 409
        await asyncio.sleep(0.8)
        assert (await claim(c, hid, "new")).status_code == 409


@pytest.mark.asyncio
async def test_complete_requires_owner_fence_and_active_lease():
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]

        # 未认领完成 409
        assert (await complete(c, hid, "w", 1)).status_code == 409
        # 交接单不存在 404
        assert (
            await c.post(
                "/handoffs/deadbeefdeadbeef/complete",
                json={"owner": "w", "fence": 1},
            )
        ).status_code == 404
        assert (
            await c.post(
                "/handoffs/deadbeefdeadbeef/claim", json={"owner": "w"}
            )
        ).status_code == 404

        assert (await claim(c, hid, "w")).status_code == 200
        # 到期后正确凭证也无法完成
        await asyncio.sleep(0.8)
        assert (await complete(c, hid, "w", 1)).status_code == 409


# ---------------------------------------------------------------------------
# 独立持久化：屏障回收不影响交接单
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handoff_survives_barrier_reclamation():
    """交接单无 TTL：屏障键被删除（模拟回收）后交接单仍可查询、接管与完成。"""
    async with await _client() as c:
        # ttl 很短：截止后很快被 Redis 回收（buffer 测试环境默认 24h，
        # 这里直接删除屏障键模拟回收时刻，验证交接单与屏障生命周期解耦）。
        bid, quorum, frozen_at = await _reached_barrier(c, ["a", "b"], ttl_s=30)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]

        redis = app.state.store.redis
        # 删除全部屏障键（模拟到期回收）
        keys = []
        async for k in redis.scan_iter(match=f"barrier:{bid}*"):
            keys.append(k)
        assert keys  # 确认确实有屏障键
        await redis.delete(*keys)

        # 屏障已不存在
        assert (await c.get(f"/barriers/{bid}")).status_code == 404
        # 不能重复建立（屏障没了 -> 404），但原交接单完整保留
        assert (await create_handoff(c, bid)).status_code == 404
        g = (await c.get(f"/handoffs/{hid}")).json()
        assert g["quorum"] == quorum
        assert g["frozen_at"] == frozen_at
        assert g["status"] == "idle"

        # 认领 -> 完成全流程不受屏障回收影响
        assert (await claim(c, hid, "finisher")).status_code == 200
        assert (await complete(c, hid, "finisher", 1)).status_code == 200
        assert (await c.get(f"/handoffs/{hid}")).json()["status"] == "completed"

        # 交接单物理键无 TTL（-1）
        for key in (f"handoff:{hid}", f"handoff:bybarrier:{bid}"):
            assert await redis.ttl(key) == -1


@pytest.mark.asyncio
async def test_handoff_quorum_is_independent_of_late_checkins():
    """建单之后的迟到报到不会改写交接单里的 quorum / frozen_at。"""
    async with await _client() as c:
        deadline = (
            datetime.now(timezone.utc) + timedelta(seconds=30)
        ).isoformat()
        r = await c.post(
            "/barriers",
            json={
                "participants": ["a", "b", "c"],
                "threshold": 2,
                "deadline": deadline,
            },
        )
        bid = r.json()["barrier_id"]
        await c.post(f"/barriers/{bid}/checkin", json={"participant_id": "a"})
        await c.post(f"/barriers/{bid}/checkin", json={"participant_id": "b"})
        frozen = (await c.get(f"/barriers/{bid}")).json()
        assert frozen["quorum"] == ["a", "b"]

        hid = (await create_handoff(c, bid)).json()["handoff_id"]

        # 建单后 c 迟到报到
        late = await c.post(f"/barriers/{bid}/checkin", json={"participant_id": "c"})
        assert late.json()["status"] == "late"

        g = (await c.get(f"/handoffs/{hid}")).json()
        assert g["quorum"] == ["a", "b"]
        assert g["frozen_at"] == frozen["frozen_at"]


# ---------------------------------------------------------------------------
# 多轮接管 fence 单调递增 & 文档
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fence_monotonic_across_multiple_takeovers():
    """空闲 -> 认领 -> 到期接管 -> 到期接管：fence 1 -> 2 -> 3 严格递增。"""
    async with await _client() as c:
        bid, _, _ = await _reached_barrier(c)
        hid = (await create_handoff(c, bid)).json()["handoff_id"]

        for round_, owner in enumerate(("w1", "w2", "w3"), start=1):
            if round_ > 1:
                await asyncio.sleep(0.8)
            r = await claim(c, hid, owner)
            assert r.status_code == 200, r.text
            assert r.json()["fence"] == round_
            assert r.json()["owner"] == owner
            # 当前持有者可续租；之后到期才丢权
            assert (
                await renew(c, hid, owner, round_)
            ).status_code == 200 if round_ < 3 else True

        g = (await c.get(f"/handoffs/{hid}")).json()
        assert g["fence"] == 3 and g["owner"] == "w3"


@pytest.mark.asyncio
async def test_handoff_endpoints_in_openapi():
    """新接口与模型出现在 OpenAPI 与根端点清单中。"""
    async with await _client() as c:
        schema = (await c.get("/openapi.json")).json()
        for path in (
            "/handoffs",
            "/handoffs/{handoff_id}",
            "/handoffs/{handoff_id}/claim",
            "/handoffs/{handoff_id}/renew",
            "/handoffs/{handoff_id}/complete",
        ):
            assert path in schema["paths"], path
        for name in (
            "CreateHandoffRequest",
            "CreateHandoffResponse",
            "HandoffView",
            "HandoffClaimRequest",
            "HandoffRenewRequest",
            "HandoffCompleteRequest",
        ):
            assert name in schema["components"]["schemas"], name

        root = (await c.get("/")).json()
        assert root["endpoints"]["create_handoff"] == "POST /handoffs"
        assert root["endpoints"]["claim_handoff"] == "POST /handoffs/{handoff_id}/claim"
