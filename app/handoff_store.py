"""Redis 持久层：达标结果交接单（handoff）与基于 fence 令牌的认领租约。

交接单独立持久保存，与屏障键空间完全分离且**不设 TTL**：屏障在截止时间
之后被回收（barrier:* 键过期删除）不影响任何已建立的交接单。

每个交接单使用如下键（``hid`` 为交接单 id，``bid`` 为屏障 id）：

``handoff:{hid}``        hash  交接单本体，永久存在：
                         barrier_id / quorum / frozen_at / created_at /
                         owner / fence / expires（当前租约到期点，Redis
                         毫秒时间戳）/ status / completed_at
``handoff:bybarrier:{bid}`` string  屏障 id -> 交接单 id 的映射，保证
                         「同一屏障重复创建返回原单」；永久存在

全部裁决均由 Redis 单节点上的 Lua 脚本串行完成，并且**一律以 Redis 自身
时钟（TIME 命令）为准**——脚本先取 Redis 毫秒时间，再做租约判定与写回，
因此应用进程时钟漂移不会影响认领 / 到期接管 / 旧凭证拒绝的结果：

- 创建：屏障不存在（meta 键已被回收）-> 404；存在但尚无 frozen 快照
  （未达标 / 已取消 / 已截止）-> 409；已达标 -> 复制 quorum / frozen_at
  建立交接单。同一屏障并发创建由映射键在同一脚本内原子去重，只有一单。
- claim：空闲（从未认领）或租约已到期才可认领；返回单调递增的 fence
  与新的租约到期点。租约仍有效时（含同 owner 重复认领）拒绝。
  接管时 fence 在旧值上 +1，旧 owner 手中的旧 fence 立即作废。
- renew / complete：必须同时携带 owner 与 fence，且租约仍在有效期内，
  否则拒绝（owner 不符 / fence 旧 / 租约已到期 / 已完成 均为 409）。
- complete：status 置 completed 为终态，此后永久拒绝再次认领、续租与
  重复完成，任何路径都不得复活。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis


# 创建交接单脚本。返回数组：
#   {'barrier_unknown'}                          屏障不存在 / 已过期回收
#   {'not_reached'}                              屏障存在但未达标冻结
#   {'replayed', hid}                            同一屏障已有交接单（幂等回放）
#   {'created', hid}                             本次新建（唯一一次写入）
#
# KEYS[3] 为新建目标 handoff:{hid}（hid 由调用方预生成）；映射键已存在时
# 脚本不触碰该目标键，直接回放既有 id。创建与去重在同一脚本内原子完成，
# 并发创建同一屏障时 Redis 串行顺序保证只有一单被写入。
CREATE_SCRIPT = """
local meta_key    = KEYS[1]
local idx_key     = KEYS[2]
local new_h_key   = KEYS[3]
local frozen_key  = KEYS[4]
local new_hid     = ARGV[1]
local now_ms      = tonumber(ARGV[2])
local bid         = ARGV[3]

if redis.call('HEXISTS', meta_key, 'threshold') == 0 then
    return {'barrier_unknown'}
end

-- 幂等去重：同一屏障只允许一份交接单，先创建者为唯一结果。
local existing = redis.call('GET', idx_key)
if existing then
    return {'replayed', existing}
end

local quorum = redis.call('HGET', frozen_key, 'quorum')
if not quorum then
    return {'not_reached'}
end

local frozen_at = redis.call('HGET', frozen_key, 'frozen_at')

-- 交接单独立持久保存：不设任何 TTL，屏障回收不影响它。
redis.call('HSET', new_h_key,
    'barrier_id', bid,
    'quorum', quorum,
    'frozen_at', tostring(frozen_at),
    'created_at', tostring(now_ms),
    'status', 'idle')
redis.call('SET', idx_key, new_hid)
return {'created', new_hid}
"""


# claim / renew / complete / view 通用：读取交接单并以 Redis 时间裁决。
#
# 认领脚本。返回数组：
#   {'unknown'}                       交接单不存在
#   {'completed'}                     已完成，永久拒绝再次认领
#   {'active', owner, expires_str}    租约仍有效，不能重复 / 抢认领
#   {'claimed', owner, fence, expires_str}  本次认领 / 到期接管成功
#
# 竞争认领与到期接管完全由 Redis 单线程串行化：「查租约 -> 判定空闲/到期
# -> fence+1、写 owner/到期点」在同一脚本内，以脚本内 TIME 取到的 Redis
# 时间为唯一时钟，两个并发认领必有且仅有一个赢家，另一个得到 active。
CLAIM_SCRIPT = """
local h_key     = KEYS[1]
local owner     = ARGV[1]
local lease_ms  = tonumber(ARGV[2])

if redis.call('HEXISTS', h_key, 'status') == 0 then
    return {'unknown'}
end
if redis.call('HGET', h_key, 'status') == 'completed' then
    return {'completed'}
end

-- 以 Redis 自身时钟裁决，不信任任何调用方传入的时间。
local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local expires = tonumber(redis.call('HGET', h_key, 'expires') or '0')
if expires > now_ms then
    -- 租约仍有效：持有者（即使与请求者同一 owner）独占，拒绝重复认领。
    return {'active', redis.call('HGET', h_key, 'owner'), tostring(expires)}
end

-- 空闲（fence 缺省 0）或租约到期：认领 / 接管。fence 单调递增，
-- 每次接管都在旧 fence 上 +1，旧 owner 的旧 fence 从此无法续租 / 完成。
local fence = tonumber(redis.call('HGET', h_key, 'fence') or '0') + 1
local new_expires = now_ms + lease_ms
redis.call('HSET', h_key,
    'owner', owner,
    'fence', tostring(fence),
    'expires', tostring(new_expires),
    'status', 'active')
return {'claimed', owner, tostring(fence), tostring(new_expires)}
"""


# 续租脚本。返回数组：
#   {'unknown'}                       交接单不存在
#   {'completed'}                     已完成，终态拒绝续租（与重复完成）
#   {'no_owner'}                      从未被认领，无凭证可续
#   {'bad_owner'}                     owner 与当前持有者不符
#   {'bad_fence', current_fence}      fence 不是当前凭证（旧接管凭证作废）
#   {'lease_expired', expires_str}    租约已到期（可能已被他人接管）
#   {'renewed', fence, expires_str}   续租成功，到期点向后顺延一个租约窗口
#
# owner / fence 校验与租约有效期判定在同一脚本内以 Redis TIME 原子裁决。
RENEW_SCRIPT = """
local h_key    = KEYS[1]
local owner    = ARGV[1]
local fence    = tonumber(ARGV[2])
local lease_ms = tonumber(ARGV[3])

if redis.call('HEXISTS', h_key, 'status') == 0 then
    return {'unknown'}
end
if redis.call('HGET', h_key, 'status') == 'completed' then
    return {'completed'}
end

local cur_owner = redis.call('HGET', h_key, 'owner')
if not cur_owner then
    return {'no_owner'}
end
if cur_owner ~= owner then
    return {'bad_owner'}
end

local cur_fence = tonumber(redis.call('HGET', h_key, 'fence'))
if cur_fence ~= fence then
    return {'bad_fence', tostring(cur_fence)}
end

local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local expires = tonumber(redis.call('HGET', h_key, 'expires') or '0')
if expires <= now_ms then
    return {'lease_expired', tostring(expires)}
end

local new_expires = now_ms + lease_ms
redis.call('HSET', h_key, 'expires', tostring(new_expires))
return {'renewed', tostring(cur_fence), tostring(new_expires)}
"""


# 完成脚本。返回数组：
#   {'unknown'}                    交接单不存在
#   {'completed'}                  已完成（重复完成 / 旧凭证完成均拒绝）
#   {'no_owner'} / {'bad_owner'} / {'bad_fence', f} / {'lease_expired', e}
#                                  与续租相同的凭证 / 租约校验
#   {'done', completed_at_str}     本次完成，终态唯一一次写入
COMPLETE_SCRIPT = """
local h_key  = KEYS[1]
local owner  = ARGV[1]
local fence  = tonumber(ARGV[2])

if redis.call('HEXISTS', h_key, 'status') == 0 then
    return {'unknown'}
end
if redis.call('HGET', h_key, 'status') == 'completed' then
    return {'completed'}
end

local cur_owner = redis.call('HGET', h_key, 'owner')
if not cur_owner then
    return {'no_owner'}
end
if cur_owner ~= owner then
    return {'bad_owner'}
end

local cur_fence = tonumber(redis.call('HGET', h_key, 'fence'))
if cur_fence ~= fence then
    return {'bad_fence', tostring(cur_fence)}
end

local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local expires = tonumber(redis.call('HGET', h_key, 'expires') or '0')
if expires <= now_ms then
    return {'lease_expired', tostring(expires)}
end

redis.call('HSET', h_key, 'status', 'completed', 'completed_at', tostring(now_ms))
return {'done', tostring(now_ms)}
"""


# 查询脚本：读取快照与租约，并以脚本内 Redis TIME 判定租约是否仍有效。
# 返回扁平数组：
#   {'unknown'}
#   {bid, quorum_json, frozen_at_s, created_at_ms, status,
#    owner_or_'', fence_or_'', expires_or_'', completed_at_or_''}
# status 为归一化后的对外状态（idle/active/expired/completed）。
VIEW_SCRIPT = """
local h_key = KEYS[1]

if redis.call('HEXISTS', h_key, 'status') == 0 then
    return {'unknown'}
end

local stored_status = redis.call('HGET', h_key, 'status')
local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local expires_raw = redis.call('HGET', h_key, 'expires')
local expires = tonumber(expires_raw or '0')
local status = stored_status
if stored_status == 'active' and expires <= now_ms then
    -- 租约到期点已过：归一化为 expired，等待接管；持有者字段保留供排查。
    status = 'expired'
end

return {
    redis.call('HGET', h_key, 'barrier_id'),
    redis.call('HGET', h_key, 'quorum'),
    redis.call('HGET', h_key, 'frozen_at'),
    redis.call('HGET', h_key, 'created_at'),
    status,
    redis.call('HGET', h_key, 'owner') or '',
    redis.call('HGET', h_key, 'fence') or '',
    expires_raw or '',
    redis.call('HGET', h_key, 'completed_at') or ''
}
"""


class HandoffNotFound(Exception):
    """交接单不存在。"""


class HandoffBarrierNotFound(Exception):
    """创建交接单时屏障不存在或已过期回收。"""


class BarrierNotReached(Exception):
    """创建交接单时屏障尚未达标（waiting / cancelled / expired）。"""


class HandoffConflict(Exception):
    """认领 / 续租 / 完成被拒绝。

    参数携带机器可判别的原因：completed / active / no_owner / bad_owner /
    bad_fence / lease_expired。
    """


@dataclass(frozen=True)
class HandoffRecord:
    handoff_id: str
    barrier_id: str
    quorum: list[str]
    # Unix 毫秒：frozen_at 由屏障冻结快照（秒，含毫秒小数）换算而来，
    # created_at / 租约到期点 / completed_at 直接取 Redis 毫秒时钟
    frozen_at_ms: int
    created_at_ms: int
    # 归一化状态：idle / active / expired / completed
    status: str
    owner: str | None
    fence: int | None
    expires_ms: int | None
    completed_at_ms: int | None

    @property
    def lease_active(self) -> bool:
        return self.status == "active"


class HandoffStore:
    def __init__(self, redis: Redis, lease_ms: int) -> None:
        self.redis = redis
        self.lease_ms = lease_ms
        self._create = redis.register_script(CREATE_SCRIPT)
        self._claim = redis.register_script(CLAIM_SCRIPT)
        self._renew = redis.register_script(RENEW_SCRIPT)
        self._complete = redis.register_script(COMPLETE_SCRIPT)
        self._view = redis.register_script(VIEW_SCRIPT)

    @staticmethod
    def handoff_key(handoff_id: str) -> str:
        return f"handoff:{handoff_id}"

    @staticmethod
    def barrier_index_key(barrier_id: str) -> str:
        return f"handoff:bybarrier:{barrier_id}"

    async def _redis_time_ms(self) -> int:
        sec, usec = await self.redis.time()
        return int(sec) * 1000 + int(usec) // 1000

    async def create(self, barrier_id: str) -> tuple[HandoffRecord, bool]:
        """为已达标屏障建立交接单；同一屏障重复创建返回原单。

        返回 ``(record, created)``：``created=True`` 为首次建立；
        ``False`` 为幂等回放既有交接单。屏障不存在 / 已回收 ->
        :class:`HandoffBarrierNotFound`；屏障存在但未达标 ->
        :class:`BarrierNotReached`。
        """
        handoff_id = uuid.uuid4().hex
        keys = [
            f"barrier:{barrier_id}",
            self.barrier_index_key(barrier_id),
            self.handoff_key(handoff_id),
            f"barrier:{barrier_id}:frozen",
        ]
        now_ms = await self._redis_time_ms()
        raw = await self._create(
            keys=keys,
            args=[handoff_id, str(now_ms), barrier_id],
        )
        status = raw[0]
        if status == "barrier_unknown":
            raise HandoffBarrierNotFound(barrier_id)
        if status == "not_reached":
            raise BarrierNotReached(barrier_id)

        existing_id = raw[1]
        created = status == "created"
        # replay 时 raw[1] 是既有交接单 id；created 时即本次预生成 id。
        record = await self.get(existing_id)
        assert record is not None
        return record, created

    async def get(self, handoff_id: str) -> HandoffRecord | None:
        raw = await self._view(keys=[self.handoff_key(handoff_id)])
        if raw[0] == "unknown":
            return None
        return self._parse(raw, handoff_id)

    @staticmethod
    def _parse(raw: list[Any], handoff_id: str) -> HandoffRecord:
        (
            barrier_id,
            quorum_json,
            frozen_at,
            created_at,
            status,
            owner,
            fence,
            expires,
            completed_at,
        ) = raw
        return HandoffRecord(
            handoff_id=handoff_id,
            barrier_id=barrier_id,
            quorum=json.loads(quorum_json),
            # 屏障 frozen 快照以「秒（含 3 位毫秒小数）」存储，换算为毫秒
            # 与交接单内其余时间戳（created_at / expires / completed_at）对齐；
            # round 抵消秒小数字符串转 double 再乘 1000 的舍入误差。
            frozen_at_ms=int(round(float(frozen_at) * 1000)),
            created_at_ms=int(created_at),
            status=status,
            owner=owner or None,
            fence=int(fence) if fence else None,
            expires_ms=int(expires) if expires else None,
            completed_at_ms=int(completed_at) if completed_at else None,
        )

    async def claim(self, handoff_id: str, owner: str) -> HandoffRecord:
        """认领 / 到期接管。空闲或租约到期才成功，返回带新 fence 的最新单据。

        租约仍有效（含同 owner 重复认领）-> :class:`HandoffConflict`（active）；
        已完成 -> :class:`HandoffConflict`（completed）；单据不存在 ->
        :class:`HandoffNotFound`。
        """
        raw = await self._claim(
            keys=[self.handoff_key(handoff_id)],
            args=[owner, str(self.lease_ms)],
        )
        status = raw[0]
        if status == "unknown":
            raise HandoffNotFound(handoff_id)
        if status in ("completed", "active"):
            raise HandoffConflict(status)

        record = await self.get(handoff_id)
        assert record is not None
        return record

    async def renew(self, handoff_id: str, owner: str, fence: int) -> HandoffRecord:
        """续租：owner + fence 必须为当前凭证且租约仍有效，否则冲突。"""
        raw = await self._renew(
            keys=[self.handoff_key(handoff_id)],
            args=[owner, str(fence), str(self.lease_ms)],
        )
        self._raise_for_mutation(handoff_id, raw)
        record = await self.get(handoff_id)
        assert record is not None
        return record

    async def complete(self, handoff_id: str, owner: str, fence: int) -> HandoffRecord:
        """完成交接：凭证有效且租约仍有效才生效，终态永久拒绝再次认领。"""
        raw = await self._complete(
            keys=[self.handoff_key(handoff_id)],
            args=[owner, str(fence)],
        )
        self._raise_for_mutation(handoff_id, raw)
        record = await self.get(handoff_id)
        assert record is not None
        return record

    @staticmethod
    def _raise_for_mutation(handoff_id: str, raw: list[Any]) -> None:
        status = raw[0]
        if status == "unknown":
            raise HandoffNotFound(handoff_id)
        if status in (
            "completed",
            "no_owner",
            "bad_owner",
            "bad_fence",
            "lease_expired",
        ):
            raise HandoffConflict(status)
        # renewed / done 为成功，由调用方回读最新单据。

    async def aclose(self) -> None:
        # 与屏障存储共用同一 Redis 连接时不在此关闭；保留方法以备独立使用。
        return None
