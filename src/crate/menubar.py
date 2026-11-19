"""Menu-bar status on macOS when PyObjC is installed. Otherwise the caller prints the same text."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable


def available() -> bool:
    return importlib.util.find_spec("AppKit") is not None


def run(poll: Callable[[], str], interval: float) -> bool:
    """Block in a status item. Returns False when AppKit is not installed."""
    if not available():
        return False
    from AppKit import (  # type: ignore[import-not-found]
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSMenu,
        NSMenuItem,
        NSStatusBar,
        NSVariableStatusItemLength,
    )
    from Foundation import NSObject, NSTimer  # type: ignore[import-not-found]

    class Poller(NSObject):  # type: ignore[misc]
        def tick_(self, _timer: object) -> None:
            self.status.setTitle_(self.poll())

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    status = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
    status.setTitle_(poll())
    menu = NSMenu.alloc().init()
    menu.addItem_(
        NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit Crate", "terminate:", "q")
    )
    status.setMenu_(menu)
    poller = Poller.alloc().init()
    poller.poll = poll
    poller.status = status
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        float(interval), poller, "tick:", None, True
    )
    app.run()
    return True
