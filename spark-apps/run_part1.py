import subprocess

def run(cmd):
    print("\nRUN:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    # Sequential: LR then RF (Part One requirement)
    run([
        "/opt/spark/bin/spark-submit",
        "--master", "spark://spark-master:7077",
        "/opt/spark-apps/train_lr.py",
        "--mode", "integrated",
        "--shared_base", "/opt/spark-shared",
        "--sample_frac", "1.0",
        "--out", "/opt/spark-apps/results/part1_lr.json"
    ])

    run([
        "/opt/spark/bin/spark-submit",
        "--master", "spark://spark-master:7077",
        "/opt/spark-apps/train_rf.py",
        "--mode", "integrated",
        "--shared_base", "/opt/spark-shared",
        "--sample_frac", "1.0",
        "--out", "/opt/spark-apps/results/part1_rf.json"
    ])

    print("\nPart One done. Check /opt/spark-apps/results/")
