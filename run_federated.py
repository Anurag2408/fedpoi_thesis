"""
08_run_federated_fixed.py
Fixed script to run federated training with better output handling
"""

import subprocess
import time
import os
import sys
from threading import Thread


def print_output(process, name):
    """Print process output in real-time"""
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"[{name}] {line.strip()}")


def run_federated_learning(n_clients=10, n_rounds=10):
    """Run federated learning experiment"""

    print("=" * 60)
    print("FEDERATED POI RECOMMENDATION - TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Clients: {n_clients}")
    print(f"  - Rounds: {n_rounds}")
    print(f"  - Server: localhost:8080")
    print("\nStarting federated learning...")
    print("=" * 60)

    processes = []
    threads = []

    try:
        # Start server
        print("\n[Main] Starting server...")
        server_process = subprocess.Popen(
            [sys.executable, "federated_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        processes.append(('server', server_process))

        # Start thread to print server output
        server_thread = Thread(target=print_output, args=(server_process, "SERVER"))
        server_thread.daemon = True
        server_thread.start()
        threads.append(server_thread)

        # Wait for server to start
        print("[Main] Waiting 5 seconds for server to initialize...")
        time.sleep(5)

        # Start clients
        print(f"\n[Main] Starting {n_clients} clients...")

        for client_id in range(n_clients):
            print(f"[Main] Starting client {client_id}...")

            client_process = subprocess.Popen(
                [sys.executable, "federated_client.py", str(client_id)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            processes.append((f'client_{client_id}', client_process))

            # Start thread to print client output
            client_thread = Thread(target=print_output, args=(client_process, f"CLIENT-{client_id}"))
            client_thread.daemon = True
            client_thread.start()
            threads.append(client_thread)

            time.sleep(1)  # Stagger starts

        print(f"\n[Main] All {n_clients} clients started!")
        print("[Main] Training in progress...")
        print("[Main] This will take 5-10 minutes. Watch the output below...\n")
        print("=" * 60)

        # Wait for all processes to complete
        for name, process in processes:
            process.wait()

        # Wait a bit for threads to finish printing
        time.sleep(2)

        print("\n" + "=" * 60)
        print("✓ FEDERATED TRAINING COMPLETE!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n[Main] Interrupted by user. Cleaning up...")

    except Exception as e:
        print(f"\n[Main] Error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        print("\n[Main] Stopping all processes...")
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass

        print("[Main] All processes stopped")


if __name__ == "__main__":
    # Change to project directory
    project_dir = '/Users/srivanur/Documents/fedpoi_thesis'
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}\n")

    # Run federated learning
    run_federated_learning(n_clients=10, n_rounds=10)