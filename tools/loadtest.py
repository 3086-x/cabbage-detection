"""HTTP 压测：给服务端报 QPS 与 P95。

为什么不用 locust
--------------
locust 要单独起进程 + 写 locustfile，对一个只需要「固定并发打 N 次、报分位」的
任务太重。更重要的是：压测客户端本身不能成为瓶颈，否则测的是客户端。
这里用固定线程数 + 复用连接的方式，把客户端开销压到可忽略。

关键指标
------
QPS          吞吐
P50/P95/P99  延迟分位。只看平均值会被长尾骗，服务端 SLO 通常按 P95 定
错误率        HTTP 非 200 或超时

跑法
----
    python tools/loadtest.py --url http://127.0.0.1:8000 --images 40 --workers 25
    python tools/loadtest.py --url http://127.0.0.1:8000 --preset batch4

产物
----
    eval_reports/loadtest_<workers>w.json
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import statistics
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_multipart(files: list[tuple[str, bytes]], field: str = "file") -> tuple[bytes, str]:
    """手工拼 multipart/form-data，避免为压测引入额外依赖。"""
    boundary = f"----cvagent{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, data in files:
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def one_request(url: str, body: bytes, ctype: str, timeout: float) -> tuple[bool, float, str]:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            ok = 200 <= resp.status < 300
            err = "" if ok else f"HTTP {resp.status}"
    except Exception as e:  # noqa: BLE001
        ok, err = False, str(e)[:120]
    return ok, (time.perf_counter() - t0) * 1000, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--images", type=int, default=40, help="参与轮转的图片数")
    ap.add_argument("--requests", type=int, default=200, help="总请求数")
    ap.add_argument("--workers", type=int, default=16, help="客户端并发数")
    ap.add_argument("--batch", type=int, default=1, help="每个请求带几张图（1=单图接口）")
    ap.add_argument("--conf", type=float, default=0.65)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--warmup", type=int, default=5, help="正式计时前的预热请求数")
    ap.add_argument("--preset", default="", choices=["", "batch4"], help="预设参数组合")
    args = ap.parse_args()

    if args.preset == "batch4":
        args.batch, args.workers = 4, 8

    img_dir = ROOT / "dataset/test/images"
    imgs = sorted(img_dir.glob("*.jpg"))[: args.images]
    if not imgs:
        raise SystemExit(f"找不到测试图片: {img_dir}")
    payloads = [(p.name, p.read_bytes()) for p in imgs]
    print(f"[info] {len(payloads)} 张图轮转，每请求 {args.batch} 张，客户端并发 {args.workers}，总请求 {args.requests}")

    endpoint = f"{args.url.rstrip('/')}/predict" if args.batch == 1 else f"{args.url.rstrip('/')}/predict/batch"
    field = "file" if args.batch == 1 else "files"
    endpoint += f"?conf={args.conf}"

    # 预先把 batch 组合的 multipart 都准备好，避免把拼包时间算进延迟
    bodies: list[tuple[bytes, str]] = []
    for i in range(0, len(payloads), args.batch):
        chunk = payloads[i : i + args.batch]
        if len(chunk) < args.batch:  # 最后不足一个 batch 的补齐
            chunk = (chunk * args.batch)[: args.batch]
        bodies.append(build_multipart(chunk, field=field))
    if not bodies:
        raise SystemExit("没有可用的请求体")

    # 预热
    with ThreadPoolExecutor(max_workers=min(4, args.workers)) as ex:
        list(ex.map(lambda _: one_request(endpoint, *bodies[0], args.timeout), range(args.warmup)))

    lat: list[float] = []
    errors: dict[str, int] = {}
    lock = threading.Lock()
    counter = {"i": 0}

    def task(_):
        with lock:
            body, ctype = bodies[counter["i"] % len(bodies)]
            counter["i"] += 1
        ok, ms, err = one_request(endpoint, body, ctype, args.timeout)
        with lock:
            lat.append(ms)
            if not ok:
                errors[err] = errors.get(err, 0) + 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(task, range(args.requests)))
    wall = time.perf_counter() - t0

    lat.sort()
    n = len(lat)
    ok_n = sum(1 for x in lat if x > 0) - sum(errors.values())

    def pct(p: float) -> float:
        if not lat:
            return 0.0
        k = min(n - 1, max(0, int(round(p / 100 * (n - 1)))))
        return lat[k]

    out = {
        "endpoint": endpoint,
        "requests": args.requests,
        "concurrency": args.workers,
        "batch": args.batch,
        "wall_s": round(wall, 2),
        "qps": round(args.requests / wall, 2),
        "ok": ok_n,
        "failed": sum(errors.values()),
        "error_detail": errors,
        "latency_ms": {
            "mean": round(statistics.fmean(lat), 2) if lat else 0,
            "p50": round(pct(50), 2),
            "p95": round(pct(95), 2),
            "p99": round(pct(99), 2),
            "max": round(max(lat), 2) if lat else 0,
        },
    }

    print(f"\nQPS={out['qps']}  成功 {out['ok']}/{args.requests}  墙钟 {out['wall_s']}s")
    print(f"延迟 ms: P50={out['latency_ms']['p50']}  P95={out['latency_ms']['p95']}  "
          f"P99={out['latency_ms']['p99']}  max={out['latency_ms']['max']}")
    if errors:
        print(f"错误: {errors}")

    Path("eval_reports").mkdir(exist_ok=True)
    tag = f"w{args.workers}_b{args.batch}"
    path = Path(f"eval_reports/loadtest_{tag}.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
