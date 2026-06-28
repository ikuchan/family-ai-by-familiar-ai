#!/usr/bin/env python3
"""
Tapo / ONVIF PTZ 絶対位置 実機確認スクリプト
==========================================

目的：このカメラが ONVIF 経由で「現在の絶対 pan/tilt 位置」を返すかを確かめる。
  - 返る（かつ動かすと値が変わる）→ T が向きをカメラから直接読める。I→T で向きを渡す必要なし。
  - 返らない / 0 固定 / 動かしても変わらない → 自分の振り履歴から角度を作る前提（I→T で向きを渡す）。

既存実装（src/familiar_agent/tools/camera.py）と同じ非同期 ONVIF API を使う。

使い方：
  アプリと同じ環境で実行する。uv で起動しているなら uv run を使う。
  このスクリプトは起動時に .env を自動読込するので（アプリ config.py と同じ挙動）、
  プロジェクト直下に置いて次を実行するだけで .env のカメラ設定を拾う：

  # 推奨（プロジェクト直下・.env を自動で読む）
  uv run python tapo_ptz_check.py

  # 読み取りのみ（カメラを動かさない）
  uv run python tapo_ptz_check.py --no-move

  ---- 以下は .env を使わず手で渡す場合 ----

  # 環境変数で渡す（アプリと同じ変数名）
  export CAMERA_HOST=192.168.0.50
  export CAMERA_USERNAME=admin
  export CAMERA_PASSWORD=yourpass
  export CAMERA_ONVIF_PORT=2020       # Tapo 既定。違うなら CAMERA_PTZ_PORT でも可
  python tapo_ptz_check.py

  # もしくは引数で
  python tapo_ptz_check.py --host 192.168.0.50 --user admin --password yourpass --port 2020

  # 読み取りのみ（カメラを動かさない）にしたい場合
  python tapo_ptz_check.py --no-move

注意：既定では「ごく小さく振って→位置が変わるか確認→振って戻す」ライブ確認をします。
      動かしたくないときは --no-move を付けてください。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# .env 自動読込：uv run（アプリと同じ環境）で起動するなら、プロジェクト直下の
# .env を拾う。python-dotenv はアプリ依存に含まれるので import できるはず。
# アプリの config.py と同じく load_dotenv() は CWD から上方向に .env を探索する。
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def _cam_env(*names: str, default: str | None = None) -> str | None:
    """先に見つかった非空の環境変数を返す。camera.py の接続パラメータ決定と同じ
    優先順位を再現するため、PTZオーバーライド → CAMERA_* → TAPO_* 別名 の順で渡す。
    （camera.py はさらに RTSP URL 内の user/pass にもフォールバックするが、診断用途
      では CAMERA_USERNAME / CAMERA_PASSWORD で足りるため、ここでは省略している。）"""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _resolve_wsdl_dir() -> str | None:
    """onvif パッケージ同梱の wsdl ディレクトリを探す。env で上書き可。"""
    env = os.environ.get("CAMERA_ONVIF_WSDL")
    if env and os.path.isdir(env):
        return env
    try:
        import onvif  # type: ignore

        cand = os.path.join(os.path.dirname(onvif.__file__), "wsdl")
        if os.path.isdir(cand):
            return cand
    except Exception:
        pass
    # アプリ内のヘルパがあれば使う
    try:
        from familiar_agent.setup import _onvif_wsdl_dir  # type: ignore

        d = _onvif_wsdl_dir()
        if d and os.path.isdir(str(d)):
            return str(d)
    except Exception:
        pass
    return None


def _pantilt(status) -> tuple[float, float] | None:
    """GetStatus 応答から PanTilt(x,y) を安全に取り出す。無ければ None。"""
    try:
        pos = getattr(status, "Position", None)
        if pos is None:
            return None
        pt = getattr(pos, "PanTilt", None)
        if pt is None:
            return None
        x = getattr(pt, "x", None)
        y = getattr(pt, "y", None)
        if x is None or y is None:
            return None
        return (float(x), float(y))
    except Exception:
        return None


def _abs_pantilt_supported(nodes) -> bool | None:
    """GetNodes から AbsolutePanTiltPositionSpace の有無を判定。判定不能なら None。"""
    try:
        for node in nodes or []:
            spaces = getattr(node, "SupportedPTZSpaces", None)
            if spaces is None:
                continue
            absspace = getattr(spaces, "AbsolutePanTiltPositionSpace", None)
            if absspace:  # 非空リスト
                return True
        return False
    except Exception:
        return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--host",
        default=_cam_env("CAMERA_PTZ_HOST", "CAMERA_HOST", "TAPO_CAMERA_HOST"),
    )
    ap.add_argument(
        "--user",
        default=_cam_env(
            "CAMERA_PTZ_USERNAME", "CAMERA_USERNAME", "TAPO_USERNAME", default="admin"
        ),
    )
    ap.add_argument(
        "--password",
        default=_cam_env("CAMERA_PTZ_PASSWORD", "CAMERA_PASSWORD", "TAPO_PASSWORD"),
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(
            _cam_env(
                "CAMERA_PTZ_PORT", "CAMERA_ONVIF_PORT", "TAPO_ONVIF_PORT", default="2020"
            )
        ),
    )
    ap.add_argument("--no-move", action="store_true", help="カメラを動かさず読み取りのみ")
    args = ap.parse_args()

    if not args.host or not args.user or args.password is None:
        print("ERROR: host/user/password が必要です（環境変数か引数で指定）。", file=sys.stderr)
        return 2

    try:
        from onvif import ONVIFCamera  # type: ignore
    except Exception as e:
        print(f"ERROR: onvif ライブラリを import できません: {e}", file=sys.stderr)
        print("ヒント: アプリと同じ仮想環境で実行してください。", file=sys.stderr)
        return 2

    wsdl_dir = _resolve_wsdl_dir()
    ports_to_try = [args.port] + [p for p in (8080, 80, 2020) if p != args.port]

    cam = None
    ptz = None
    token = None
    cfg_token = None
    used_port = None
    last_err = None

    for port in ports_to_try:
        try:
            kwargs = {"wsdl_dir": wsdl_dir} if wsdl_dir else {}
            cam = ONVIFCamera(args.host, port, args.user, args.password, **kwargs)
            await cam.update_xaddrs()
            media = await cam.create_media_service()
            profiles = await media.GetProfiles()
            if not profiles:
                raise RuntimeError("プロファイルが空")
            token = profiles[0].token
            try:
                cfg_token = profiles[0].PTZConfiguration.token
            except Exception:
                cfg_token = None
            ptz = await cam.create_ptz_service()
            used_port = port
            break
        except Exception as e:
            last_err = e
            cam = None
            continue

    print("=" * 60)
    if cam is None or ptz is None:
        print("接続: 失敗")
        print(f"  試したポート: {ports_to_try}")
        print(f"  最後のエラー: {last_err}")
        print("  → ONVIF PTZ に接続できません。ホスト/認証/ポートを確認してください。")
        return 1

    print(f"接続: 成功 (port={used_port})")
    print(f"  ProfileToken      = {token}")
    print(f"  ConfigurationToken= {cfg_token}")

    # --- 絶対位置サポートの有無 ---
    abs_supported = None
    try:
        nodes = await ptz.GetNodes()
        abs_supported = _abs_pantilt_supported(nodes)
    except Exception as e:
        print(f"  GetNodes 失敗: {e}")
    print(f"  AbsolutePanTilt 対応 = {abs_supported}")

    # --- GetStatus（現在位置）---
    pos1 = None
    try:
        status = await ptz.GetStatus({"ProfileToken": token})
        pos1 = _pantilt(status)
        move_status = getattr(getattr(status, "MoveStatus", None), "PanTilt", None)
        utc = getattr(status, "UtcTime", None)
        print(f"  GetStatus Position(PanTilt) = {pos1}")
        print(f"  GetStatus MoveStatus.PanTilt= {move_status}")
        print(f"  GetStatus UtcTime           = {utc}")
    except Exception as e:
        print(f"  GetStatus 失敗: {e}")

    # --- ライブ確認（小さく振って位置が変わるか）---
    pos2 = None
    if not args.no_move:
        try:
            print("  [move] 小さく右に振ります (x=-0.05) ...")
            await ptz.RelativeMove(
                {"ProfileToken": token, "Translation": {"PanTilt": {"x": -0.05, "y": 0.0}}}
            )
            await asyncio.sleep(1.0)
            status2 = await ptz.GetStatus({"ProfileToken": token})
            pos2 = _pantilt(status2)
            print(f"  振った後の Position(PanTilt) = {pos2}")
            print("  [move] 元に戻します (x=+0.05) ...")
            await ptz.RelativeMove(
                {"ProfileToken": token, "Translation": {"PanTilt": {"x": 0.05, "y": 0.0}}}
            )
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f"  move/GetStatus 失敗: {e}")
    else:
        print("  [move] --no-move 指定のためライブ確認はスキップ")

    # --- 判定 ---
    print("-" * 60)
    live = pos1 is not None and pos2 is not None and pos1 != pos2
    static_zero = pos1 is not None and pos1 == (0.0, 0.0) and (pos2 is None or pos2 == (0.0, 0.0))

    if live:
        verdict = (
            "OK: 絶対位置が取得でき、動かすと値が変わる（ライブ）。\n"
            "    → T がカメラから直接 向きを読める。I→T で向きを渡す必要なし。"
        )
    elif pos1 is not None and not static_zero and args.no_move:
        verdict = (
            "MAYBE: 位置は返るが、ライブ確認をしていない（--no-move）。\n"
            "    → --no-move を外して、動かして値が変わるか確認してください。"
        )
    elif pos1 is None:
        verdict = (
            "NG: GetStatus に Position が含まれない。\n"
            "    → 絶対位置は取れない。自分の振り履歴から角度を作る（I→T で向きを渡す）。"
        )
    else:
        verdict = (
            "NG: 位置は返るが 0 固定/動かしても変わらない（使えない）。\n"
            "    → 自分の振り履歴から角度を作る（I→T で向きを渡す）。"
        )

    print("判定:")
    print("  " + verdict.replace("\n", "\n  "))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
