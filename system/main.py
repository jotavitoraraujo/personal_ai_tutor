### --- IMPORTS --- ###
import asyncio
import logging as log

from pynput import keyboard

from system.core.linguist_conductor import LinguistConductor
from system.settings_log import log_system
from system.utils.config import Config


async def main() -> None:
    log_system()
    log.info("[SYSTEM] Initializing Linguist AI Tutor...")

    config = Config()
    conductor = LinguistConductor(config)

    trigger_key = keyboard.Key.f8

    listener = keyboard.Listener(
        on_press=lambda key: conductor.on_press() if key == trigger_key else None,
        on_release=lambda key: conductor.on_release() if key == trigger_key else None,
    )

    log.info(f"[CONTROL] Ready! Hold '{trigger_key.name.upper()}' to talk.")
    listener.start()

    try:
        await conductor.run()
    except KeyboardInterrupt:
        log.warning("\n[SYSTEM] Shutdown signal received.")
    finally:
        listener.stop()
        log.info("[SYSTEM] Clean exit accomplished.")


if __name__ == "__main__":
    asyncio.run(main())
