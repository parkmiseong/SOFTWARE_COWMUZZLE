'''
소의 코지문 인식 시스템 구현
'''

import torch
import torch.nn as nn

class CowFingerprintRecognitionSystem:
    def __init__(self):
        self.database = {}  # 소의 ID와 코지문 데이터를 저장하는 데이터베이스

    def enroll_cow(self, cow_id, fingerprint_data):
        """소의 ID와 코지문 데이터를 데이터베이스에 등록"""
        self.database[cow_id] = fingerprint_data
        print(f"소 ID {cow_id}이(가) 등록되었습니다.")

    def recognize_cow(self, fingerprint_data):
        """주어진 코지문 데이터로 소를 인식"""
        for cow_id, stored_fingerprint in self.database.items():
            if self.compare_fingerprints(fingerprint_data, stored_fingerprint):
                print(f"소 ID {cow_id}이(가) 인식되었습니다.")
                return cow_id
        print("인식된 소가 없습니다.")
        return None

    def compare_fingerprints(self, fp1, fp2):
        """코지문 데이터를 비교하는 간단한 알고리즘 (예시)"""
        # 실제 구현에서는 더 복잡한 알고리즘이 필요할 수 있습니다.
        return fp1 == fp2