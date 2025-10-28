import platform
import os
import time

def reboot_system():
    """
    Detects the current operating system and executes the appropriate
    reboot command.
    Asks for user confirmation before proceeding.
    """
    
    # 1. Detect Operating System
    current_os = platform.system()
    command = ""

    if current_os == "Windows":
        print(f"Operating System detected: Windows")
        # Windows: /r (reboot), /t 0 (immediately)
        command = "shutdown /r /t 0"
    elif current_os == "Linux":
        print(f"Operating System detected: Linux (Ubuntu-like)")
        # Linux: -r (reboot), now (immediately)
        # This command requires root privileges.
        command = "shutdown -r now"
    else:
        print(f"Unsupported Operating System: {current_os}")
        return

    # 2. User Confirmation (Important!)
    # This prevents accidental reboots.
    try:
        confirm = input("Are you sure you want to reboot the system? (y/n): ").strip().lower()
    except EOFError:
        confirm = 'n' # Treat empty input (like in a non-interactive pipe) as 'no'

    if confirm == 'y' or confirm == 'yes':
        print("\n[WARNING] System will reboot in 5 seconds.")
        print("Press Ctrl+C to cancel...")
        
        try:
            # 5-second delay (provides a window to cancel)
            time.sleep(5)
            print(f"Executing reboot command: {command}")
            os.system(command)
        except KeyboardInterrupt:
            print("\nReboot has been cancelled by the user.")
    else:
        print("Reboot cancelled.")

if __name__ == "__main__":
    print("--- System Reboot Script ---")
    print("!! WARNING: This script must be run with Administrator/root privileges. !!\n")
    reboot_system()
