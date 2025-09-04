from dataset import make_example, save_dataset
from config import NUM_SAMPLES, MIN_LEN, MAX_LEN

def main():
    samples = [make_example(MIN_LEN, MAX_LEN) for _ in range(NUM_SAMPLES)]
    save_dataset(samples, "digits_dataset.json")
    print(f"Saved {len(samples)} samples to digits_dataset.json")

if __name__ == "__main__":
    main()