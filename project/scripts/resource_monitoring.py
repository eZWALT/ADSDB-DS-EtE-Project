import psutil
import time
import os
from colorama import Fore, Style, init

init(autoreset=True) 

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_percentage(value):
    return f"{value:.2f}%"

def format_bytes(size):
    if size < 1024:
        return f"{size} Bytes"
    elif size < 1024 ** 2:
        return f"{size / 1024:.2f} KB"
    elif size < 1024 ** 3:
        return f"{size / 1024 ** 2:.2f} MB"
    else:
        return f"{size / 1024 ** 3:.2f} GB"

"""Monitor CPU, memory, and disk usage with a formatted output."""
def monitor_resources(interval=1):
    try:
        while True:
            clear_console()

            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.5)
            # Get memory usage
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.used  
            total_memory = memory_info.total  
            # Get disk usage
            disk_info = psutil.disk_usage('/')
            disk_usage = disk_info.used  
            total_disk = disk_info.total   

            # Get system uptime
            uptime = time.time() - psutil.boot_time()
            uptime_hours, remainder = divmod(uptime, 3600)
            uptime_minutes, _ = divmod(remainder, 60)

            # display
            print("=" * 50)
            print(" " * 15 + Fore.CYAN + "System Resource Monitor")
            print("=" * 50)
            print(f"{'Uptime:':<15} {int(uptime_hours)} hours {int(uptime_minutes)} minutes")
            print(f"{Fore.YELLOW}{'CPU Usage:':<15} {format_percentage(cpu_usage)}")
            print(f"{Fore.GREEN}{'Memory Usage:':<15} {format_bytes(memory_usage)} / {format_bytes(total_memory)} ({format_percentage((memory_usage / total_memory) * 100)})")
            print(f"{Fore.RED}{'Disk Usage:':<15} {format_bytes(disk_usage)} / {format_bytes(total_disk)} ({format_percentage((disk_usage / total_disk) * 100)})")
            print("=" * 50)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    monitor_resources(3)
