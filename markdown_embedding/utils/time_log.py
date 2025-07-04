import time
from datetime import timedelta


class TimeLogger:
    def __init__(self, ):
        self.start_time = time.time()
        self.avg_time = 0
        self.count = 0
    
    def reset_time(self, ):
        self.start_time = time.time()
    
    def log(self, ):
        duration = time.time() - self.start_time
        spent_time = self.avg_time * self.count + duration
        self.avg_time = spent_time / (self.count + 1)
        self.count += 1
        
        print(
            f"[File {self.count}] "
            f"This: {self._format_time(duration)} | "
            f"Avg: {self._format_time(self.avg_time)} | "
            f"Lasted: {self._format_time(spent_time)}"
        )
        self.start_time = time.time()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """将秒转换为 HH:MM:SS 格式"""
        return str(timedelta(seconds=int(seconds))).split('.')[0]