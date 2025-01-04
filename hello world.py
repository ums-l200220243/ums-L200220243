
from metaflow import FlowSpec, step

class helo(FlowSpec):

    @step
    def start(self):
        print("Hello World!")
        self.next(self.end)

    @step
    def end(self):
        print("Alur selesai!")

if __name__ == '__main__':
    # Jalankan alur hanya jika skrip dijalankan sebagai program utama
    flow = helo()
    flow.run()
