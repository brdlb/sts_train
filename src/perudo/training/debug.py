"""
Debug mode management for training.

Provides thread-safe debug mode control with keyboard shortcuts.
"""

import threading
import logging
from typing import Optional

# Setup logger for this module
logger = logging.getLogger(__name__)


class DebugModeManager:
    """
    Thread-safe debug mode manager.
    
    Manages debug mode state and keyboard listener for toggling debug mode
    during training (Ctrl+D shortcut).
    """
    
    def __init__(self):
        """Initialize debug mode manager."""
        self._debug_mode = threading.Event()
        self._debug_mode.clear()  # Disable debug mode by default
        self._lock = threading.Lock()
        self._listener_thread: Optional[threading.Thread] = None
    
    def get(self) -> bool:
        """
        Get current debug mode state (thread-safe).
        
        Returns:
            True if debug mode is enabled, False otherwise
        """
        with self._lock:
            return self._debug_mode.is_set()
    
    def set(self, enabled: bool) -> None:
        """
        Set debug mode state (thread-safe).
        
        Args:
            enabled: True to enable debug mode, False to disable
        """
        with self._lock:
            if enabled:
                self._debug_mode.set()
                logger.info("\n[DEBUG MODE] ON - Every move in the game will be printed")
            else:
                self._debug_mode.clear()
                logger.info("\n[DEBUG MODE] OFF - Training continues in normal mode")
    
    @property
    def debug_mode_event(self) -> threading.Event:
        """
        Get the underlying Event object for direct access.
        
        Returns:
            The threading.Event object used for debug mode state
        """
        return self._debug_mode
    
    def start_keyboard_listener(self) -> None:
        """
        Start keyboard listener thread for toggling debug mode.
        
        Listens for Ctrl+D to toggle debug mode. Runs in a daemon thread
        so it doesn't block program exit.
        """
        if self._listener_thread is not None and self._listener_thread.is_alive():
            # Listener already running
            return
        
        def listener_target():
            """Thread function to listen for keyboard shortcuts (Ctrl+D to toggle debug mode)."""
            try:
                from pynput import keyboard
                
                # Track last toggle time to prevent rapid toggling
                last_toggle_time = [0.0]
                toggle_lock = threading.Lock()
                
                def toggle_debug_mode():
                    """Toggle debug mode when Ctrl+D is pressed."""
                    try:
                        import time
                        current_time = time.time()
                        
                        # Prevent rapid toggling (minimum 0.3 seconds between toggles)
                        with toggle_lock:
                            if current_time - last_toggle_time[0] < 0.3:
                                return
                            last_toggle_time[0] = current_time
                        
                        current_state = self.get()
                        self.set(not current_state)
                    except Exception:
                        # Silently ignore errors to prevent crashes
                        pass
                
                # Use GlobalHotKeys for better cross-platform support
                # This is the recommended way to handle global hotkeys
                hotkey_string = '<ctrl>+d'
                
                # Create GlobalHotKeys listener
                def on_activate():
                    """Called when hotkey is pressed."""
                    toggle_debug_mode()
                
                # Try using GlobalHotKeys first (more reliable)
                try:
                    logger.info("[DEBUG MODE] Keyboard listener started. Press Ctrl+D to toggle debug mode.")
                    with keyboard.GlobalHotKeys({hotkey_string: on_activate}) as listener:
                        listener.join()
                except Exception as e:
                    # If GlobalHotKeys fails, try manual tracking
                    logger.warning(f"[DEBUG MODE] GlobalHotKeys failed, using fallback method. Error: {e}")
                    # Fallback to manual tracking if GlobalHotKeys doesn't work
                    pressed_keys = set()
                    pressed_keys_lock = threading.Lock()
                    
                    def on_press(key):
                        """Handle key press events."""
                        try:
                            with pressed_keys_lock:
                                # Track Ctrl key (both left and right)
                                if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.ctrl):
                                    pressed_keys.add('ctrl')
                                # Track 'd' key
                                elif hasattr(key, 'char') and key.char and key.char.lower() == 'd':
                                    pressed_keys.add('d')
                                elif hasattr(key, 'name') and key.name and key.name.lower() == 'd':
                                    pressed_keys.add('d')
                                
                                # Check if Ctrl+D is pressed
                                if 'ctrl' in pressed_keys and 'd' in pressed_keys:
                                    toggle_debug_mode()
                                    # Clear to prevent multiple toggles
                                    pressed_keys.clear()
                        except Exception:
                            pass
                    
                    def on_release(key):
                        """Handle key release events."""
                        try:
                            with pressed_keys_lock:
                                # Remove released keys from tracking
                                if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.ctrl):
                                    pressed_keys.discard('ctrl')
                                elif hasattr(key, 'char') and key.char and key.char.lower() == 'd':
                                    pressed_keys.discard('d')
                                elif hasattr(key, 'name') and key.name and key.name.lower() == 'd':
                                    pressed_keys.discard('d')
                        except Exception:
                            pass
                    
                    # Start listening to keyboard events
                    logger.info("[DEBUG MODE] Fallback keyboard listener started. Press Ctrl+D to toggle debug mode.")
                    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                        listener.join()
            
            except ImportError:
                # If pynput is not available, use fallback method with stdin
                logger.warning("pynput not available. Using stdin-based debug mode toggle.")
                logger.info("Press 'd' + Enter to toggle debug mode, or Ctrl+C to exit.")
                
                import sys
                import time
                
                # Use a separate thread for stdin reading to avoid blocking
                def stdin_reader():
                    """Read stdin in a separate thread."""
                    while True:
                        try:
                            line = sys.stdin.readline().strip().lower()
                            if line == 'd':
                                current_state = self.get()
                                self.set(not current_state)
                        except (KeyboardInterrupt, EOFError):
                            break
                        except Exception:
                            # Silently ignore errors to prevent crashes
                            time.sleep(0.1)
                
                # Start stdin reader thread
                stdin_thread = threading.Thread(target=stdin_reader, daemon=True)
                stdin_thread.start()
                
                # Keep main thread alive
                while True:
                    time.sleep(1)
            except Exception:
                # Silently ignore errors to prevent crashes
                pass
        
        # Start listener in daemon thread
        self._listener_thread = threading.Thread(target=listener_target, daemon=True)
        self._listener_thread.start()


# Global instance for backward compatibility
# This allows existing code to use get_debug_mode() and set_debug_mode()
# without refactoring all call sites immediately
_global_manager = DebugModeManager()


def get_debug_mode() -> bool:
    """
    Get current debug mode state (thread-safe).
    
    Backward compatibility function. Uses global DebugModeManager instance.
    
    Returns:
        True if debug mode is enabled, False otherwise
    """
    return _global_manager.get()


def set_debug_mode(enabled: bool) -> None:
    """
    Set debug mode state (thread-safe).
    
    Backward compatibility function. Uses global DebugModeManager instance.
    
    Args:
        enabled: True to enable debug mode, False to disable
    """
    _global_manager.set(enabled)

