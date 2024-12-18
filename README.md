# Find_Best_Assignment

## 소개

`Find_Best_Assignment` 프로젝트는 다양한 알고리즘을 활용하여 최적의 작업 할당을 찾는 것을 목표로 합니다. 이 프로젝트는 전수 조사, 유전 알고리즘, 강화 학습 등을 포함한 여러 접근 방식을 구현하여 단일 기계 스케줄링 문제를 해결합니다.

## 주요 파일 및 디렉토리

- **`FullEnumeration.py`**: 모든 가능한 작업 할당을 탐색하여 최적의 솔루션을 찾는 전수 조사 알고리즘을 구현한 스크립트입니다.

- **`FullEnumeration_consideration.py`**: 전수 조사 알고리즘의 변형으로, 특정 제약 조건을 고려하여 최적의 작업 할당을 찾습니다.

- **`genetic_algorithm.py`**: 유전 알고리즘을 사용하여 최적의 작업 할당을 찾는 스크립트입니다.

- **`NUM_genetic_algorithm.py`**: 유전 알고리즘의 또 다른 구현으로, 특정한 수치적 접근 방식을 사용합니다.

- **`reinforcement_single_machine.py`**: 강화 학습을 활용하여 단일 기계 스케줄링 문제를 해결하는 스크립트입니다.

- **`data_generator.py`**: 알고리즘 테스트를 위한 샘플 데이터를 생성하는 유틸리티 스크립트입니다.

- **`examdata.csv`**: 테스트 및 검증을 위한 예제 데이터셋입니다.

- **`Single_Machine_Scheduling_1.pdf`** 및 **`Single_Machine_Scheduling_2_.pdf`**: 단일 기계 스케줄링 문제와 관련된 참고 문서입니다.

## 요구 사항

이 프로젝트는 Python 3.x 버전에서 실행되며, 추가로 다음과 같은 라이브러리가 필요합니다:

- numpy
- pandas
- matplotlib

필요한 라이브러리를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install numpy pandas matplotlib
