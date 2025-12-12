"""
AI Hub 추가 데이터셋 통합 스크립트

목적: Competition 56개 클래스의 데이터 불균형 해소
제약: TS2, TL2 데이터셋 사용 금지
사용 가능: TL_1, TL_3, TL_4, TS1, TS3, ... (자동 탐색)

기능:
1. aihubshell을 사용한 API 다운로드
2. 어노테이션 우선 다운로드 및 필터링
3. 필터링된 데이터만 이미지 추출
"""

import os
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional
import pandas as pd
from tqdm import tqdm


class AIHubDownloader:
    """aihubshell을 사용한 AI Hub 데이터셋 다운로드"""

    # 의약품 이미지 데이터셋 번호
    DRUG_DATASET_KEY = "576"

    def __init__(self, project_root: str, api_key: str):
        self.project_root = Path(project_root)
        self.api_key = api_key
        self.aihubshell_path = self.project_root / "aihubshell"
        self.download_dir = self.project_root / "data" / "aihub_downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def install_aihubshell(self) -> bool:
        """aihubshell 다운로드 및 설치"""
        if self.aihubshell_path.exists():
            print(f"✅ aihubshell 이미 설치됨: {self.aihubshell_path}")
            return True

        print("\n=== aihubshell 설치 ===")
        try:
            # 다운로드
            result = subprocess.run(
                ["curl", "-o", str(self.aihubshell_path),
                 "https://api.aihub.or.kr/api/aihubshell.do"],
                capture_output=True,
                text=True,
                check=True
            )

            # 실행 권한 부여
            self.aihubshell_path.chmod(0o755)

            print(f"✅ aihubshell 설치 완료: {self.aihubshell_path}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ aihubshell 설치 실패: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False

    def list_dataset_files(self) -> Optional[Dict]:
        """데이터셋 파일 목록 조회"""
        if not self.aihubshell_path.exists():
            print("❌ aihubshell이 설치되지 않았습니다.")
            return None

        print(f"\n=== 데이터셋 {self.DRUG_DATASET_KEY} 파일 목록 조회 ===")

        try:
            result = subprocess.run(
                [str(self.aihubshell_path),
                 "-mode", "l",
                 "-datasetkey", self.DRUG_DATASET_KEY],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                check=True
            )

            print(result.stdout)
            return {"stdout": result.stdout, "stderr": result.stderr}

        except subprocess.CalledProcessError as e:
            print(f"❌ 목록 조회 실패: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return None

    def download_dataset_file(self, filekey: str, dataset_name: str) -> bool:
        """
        특정 파일 다운로드

        Args:
            filekey: 다운로드할 파일의 키
            dataset_name: 데이터셋 이름 (예: TL_1, TS3)
        """
        if not self.aihubshell_path.exists():
            print("❌ aihubshell이 설치되지 않았습니다.")
            return False

        output_dir = self.download_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {dataset_name} (filekey: {filekey}) 다운로드 ===")

        try:
            result = subprocess.run(
                [str(self.aihubshell_path),
                 "-mode", "d",
                 "-datasetkey", self.DRUG_DATASET_KEY,
                 "-filekey", filekey,
                 "-aihubapikey", self.api_key],
                capture_output=True,
                text=True,
                cwd=str(output_dir),
                check=True,
                timeout=3600  # 1시간 타임아웃
            )

            print(f"✅ {dataset_name} 다운로드 완료")
            print(result.stdout)
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ {dataset_name} 다운로드 실패: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print(f"❌ {dataset_name} 다운로드 타임아웃")
            return False

    def download_all_datasets(self, exclude_datasets: Set[str] = None) -> Dict[str, Path]:
        """
        모든 데이터셋 다운로드 (TS2, TL2 제외)

        Returns:
            다운로드된 데이터셋 경로 딕셔너리
        """
        if exclude_datasets is None:
            exclude_datasets = {'TS2', 'TL2', 'TL_2', 'TS_2'}

        # TODO: 실제 filekey 목록은 list_dataset_files() 결과를 파싱해서 얻어야 함
        # 현재는 수동으로 지정
        print("\n⚠️ 수동으로 filekey를 지정해야 합니다.")
        print("먼저 list_dataset_files()를 실행하여 filekey를 확인하세요.")

        return {}


class AIHubDataIntegrator:
    """AI Hub 데이터셋을 Competition 형식에 맞춰 통합"""

    # Competition 56개 클래스 ID (문자열)
    TARGET_CLASSES = {
        '1899', '2482', '3350', '3482', '3543', '3742', '3831', '4542',
        '12080', '12246', '12777', '13394', '13899', '16231', '16261', '16547',
        '16550', '16687', '18146', '18356', '19231', '19551', '19606', '19860',
        '20013', '20237', '20876', '21324', '21770', '22073', '22346', '22361',
        '24849', '25366', '25437', '25468', '27732', '27776', '27925', '27992',
        '28762', '29344', '29450', '29666', '30307', '31862', '31884', '32309',
        '33008', '33207', '33879', '34596', '35205', '36636', '38161', '41767'
    }

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.raw_train_anno_dir = self.project_root / "data" / "train_annotations"
        self.raw_train_img_dir = self.project_root / "data" / "train_images"

        # 추가 데이터 저장 위치
        self.added_data_root = self.project_root / "data" / "added"
        self.integrated_anno_dir = self.project_root / "data" / "train_annotations_integrated"
        self.integrated_img_dir = self.project_root / "data" / "train_images_integrated"

    def analyze_current_data(self) -> Dict[str, int]:
        """현재 Competition 데이터의 클래스별 이미지 수 분석"""
        print("\n=== 1. 현재 데이터 분석 ===")

        class_counts = defaultdict(int)
        json_files = list(self.raw_train_anno_dir.rglob("*.json"))

        print(f"발견된 JSON 파일: {len(json_files)}개")

        for json_path in tqdm(json_files, desc="현재 데이터 분석"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'categories' in data and data['categories']:
                    cat_id = str(data['categories'][0]['id'])
                    if cat_id in self.TARGET_CLASSES:
                        class_counts[cat_id] += 1

            except Exception as e:
                print(f"경고: {json_path} 처리 실패 - {e}")
                continue

        # 결과 출력
        df = pd.DataFrame([
            {'class_id': k, 'count': v}
            for k, v in sorted(class_counts.items(), key=lambda x: x[1])
        ])

        print("\n클래스별 현재 데이터 수:")
        print(df.to_string(index=False))
        print(f"\n총 클래스 수: {len(class_counts)}/56")
        print(f"평균 이미지 수: {df['count'].mean():.1f}")
        print(f"최소: {df['count'].min()}, 최대: {df['count'].max()}")

        # 부족한 클래스 식별
        lacking_classes = {k: v for k, v in class_counts.items() if v < 10}
        print(f"\n⚠️ 10개 미만 클래스: {len(lacking_classes)}개")

        return dict(class_counts)

    def find_json_files_recursive(self, root_dir: Path) -> List[Path]:
        """재귀적으로 JSON 파일 찾기"""
        return list(root_dir.rglob("*.json"))

    def fix_aihub_json(self, json_path: Path, output_path: Path) -> bool:
        """
        AI Hub JSON을 Competition 형식으로 수정

        AI Hub: category_id=1, name="Drug"
        Competition: category_id=dl_idx, name=dl_name
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # images 섹션에서 dl_idx, dl_name 추출
            if 'images' not in data or not data['images']:
                return False

            img_info = data['images'][0]
            dl_idx = img_info.get('dl_idx')
            dl_name = img_info.get('dl_name')

            if not dl_idx or not dl_name:
                return False

            # dl_idx가 56개 클래스에 없으면 스킵
            if str(dl_idx) not in self.TARGET_CLASSES:
                return False

            cat_id = int(dl_idx)

            # categories 섹션 수정
            data['categories'] = [{
                'supercategory': 'pill',
                'id': cat_id,
                'name': dl_name
            }]

            # annotations 섹션의 category_id 수정
            if 'annotations' in data:
                for anno in data['annotations']:
                    anno['category_id'] = cat_id

            # 저장
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"경고: {json_path} 수정 실패 - {e}")
            return False

    def check_image_exists(self, json_path: Path, source_img_dir: Path) -> bool:
        """JSON에 해당하는 이미지 파일이 실제 존재하는지 확인"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'images' not in data or not data['images']:
                return False

            file_name = data['images'][0].get('file_name')
            if not file_name:
                return False

            # 이미지 경로 구성 (AI Hub 구조: K-조합_json/K-약코드/)
            json_rel_path = json_path.relative_to(json_path.parents[2])
            combo_name = json_path.parent.parent.name.replace('_json', '')

            img_path = source_img_dir / combo_name / file_name

            return img_path.exists()

        except Exception:
            return False

    def process_aihub_dataset(
        self,
        dataset_name: str,
        source_anno_dir: Path,
        source_img_dir: Path,
        current_class_counts: Dict[str, int],
        min_samples: int = 5,
        max_samples_per_class: Optional[int] = None
    ) -> Dict[str, int]:
        """
        AI Hub 데이터셋 처리

        Args:
            dataset_name: 데이터셋 이름 (TL_1, TL_3 등)
            source_anno_dir: 원본 어노테이션 디렉토리
            source_img_dir: 원본 이미지 디렉토리
            current_class_counts: 현재 클래스별 개수
            min_samples: 이 개수 미만인 클래스만 추가
            max_samples_per_class: 클래스당 최대 추가 개수 (None=무제한)
        """
        print(f"\n=== 2. {dataset_name} 데이터셋 처리 ===")

        if not source_anno_dir.exists():
            print(f"경고: {source_anno_dir} 디렉토리가 없습니다. 건너뜀.")
            return {}

        # 출력 디렉토리
        fixed_anno_dir = self.added_data_root / dataset_name / "annotations_fixed"
        fixed_anno_dir.mkdir(parents=True, exist_ok=True)

        # 1단계: JSON 파일 찾기
        json_files = self.find_json_files_recursive(source_anno_dir)
        print(f"발견된 JSON: {len(json_files)}개")

        # 2단계: 이미지 존재 여부 체크 + JSON 수정
        valid_jsons = []
        added_class_counts = defaultdict(int)

        for json_path in tqdm(json_files, desc=f"{dataset_name} JSON 처리"):
            # 이미지 존재 확인
            if not self.check_image_exists(json_path, source_img_dir):
                continue

            # JSON 수정
            rel_path = json_path.relative_to(source_anno_dir)
            output_path = fixed_anno_dir / rel_path

            if self.fix_aihub_json(json_path, output_path):
                # 수정된 JSON에서 클래스 ID 확인
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'categories' in data and data['categories']:
                    cat_id = str(data['categories'][0]['id'])
                    current_count = current_class_counts.get(cat_id, 0)
                    added_count = added_class_counts[cat_id]

                    # 필터링 조건
                    should_add = current_count < min_samples

                    # max_samples_per_class가 None이 아니면 제한 적용
                    if max_samples_per_class is not None:
                        should_add = should_add and added_count < max_samples_per_class

                    if should_add:
                        valid_jsons.append((json_path, output_path, cat_id))
                        added_class_counts[cat_id] += 1

        print(f"✅ 유효한 JSON: {len(valid_jsons)}개")
        print(f"✅ 추가될 클래스 수: {len(added_class_counts)}개")

        # 클래스별 추가 개수 출력
        if added_class_counts:
            df = pd.DataFrame([
                {'class_id': k, 'added': v}
                for k, v in sorted(added_class_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            print("\n클래스별 추가 데이터:")
            print(df.to_string(index=False))

        # 3단계: 이미지 복사
        print(f"\n이미지 복사 중...")
        copied_images = 0

        for src_json, fixed_json, cat_id in tqdm(valid_jsons, desc="이미지 복사"):
            try:
                with open(fixed_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                file_name = data['images'][0]['file_name']
                combo_name = src_json.parent.parent.name.replace('_json', '')

                src_img = source_img_dir / combo_name / file_name
                dst_img = self.integrated_img_dir / file_name

                if src_img.exists() and not dst_img.exists():
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_img, dst_img)
                    copied_images += 1

            except Exception as e:
                print(f"이미지 복사 실패: {e}")
                continue

        print(f"✅ 복사된 이미지: {copied_images}개")

        return dict(added_class_counts)

    def discover_available_datasets(self, specific_datasets: List[str] = None) -> Dict[str, Dict[str, Path]]:
        """
        data/added/ 디렉토리에서 사용 가능한 TL/TS 데이터셋 자동 탐색

        Args:
            specific_datasets: 특정 데이터셋만 사용 (예: ['TL_1', 'TL_3'])
                              None이면 모든 데이터셋 탐색

        Returns:
            딕셔너리 {데이터셋명: {'anno': Path, 'img': Path}}
        """
        # TS2, TL2 제외 (금지된 데이터셋)
        FORBIDDEN = {'ts2', 'tl2', 'tl_2', 'ts_2'}

        datasets = {}

        if not self.added_data_root.exists():
            print(f"경고: {self.added_data_root} 디렉토리가 없습니다.")
            return datasets

        # data/added/ 하위의 모든 디렉토리 탐색
        for subdir in sorted(self.added_data_root.iterdir()):
            if not subdir.is_dir():
                continue

            dir_name = subdir.name.lower()

            # 금지된 데이터셋 필터링
            if dir_name in FORBIDDEN:
                print(f"⚠️ {subdir.name} 제외 (사용 금지 데이터셋)")
                continue

            # TL/TS 패턴이 아니면 스킵
            if not (dir_name.startswith('tl') or dir_name.startswith('ts')):
                continue

            # 특정 데이터셋 지정된 경우 필터링
            if specific_datasets:
                # TL_1, tl1, TL1 등 다양한 형식 허용
                normalized_name = dir_name.replace('_', '').upper()
                specific_normalized = [s.replace('_', '').upper() for s in specific_datasets]
                if normalized_name not in specific_normalized:
                    continue

            # 어노테이션과 이미지 디렉토리 확인
            anno_dir = subdir / 'train_annotations'
            img_dir = subdir / 'train_images'

            if anno_dir.exists() and img_dir.exists():
                dataset_name = subdir.name.upper().replace('_', '_')  # TL1 → TL_1
                if '_' not in dataset_name and len(dataset_name) > 2:
                    # TL1 → TL_1, TS3 → TS_3
                    dataset_name = f"{dataset_name[:2]}_{dataset_name[2:]}"

                datasets[dataset_name] = {
                    'anno': anno_dir,
                    'img': img_dir
                }
                print(f"✅ 발견: {dataset_name} ({anno_dir.parent})")
            else:
                print(f"⚠️ {subdir.name} 스킵 (train_annotations 또는 train_images 없음)")

        return datasets

    def integrate_all_data(
        self,
        specific_datasets: List[str] = None,
        min_samples: int = 10,
        max_samples_per_class: Optional[int] = None
    ):
        """
        모든 데이터셋 통합

        Args:
            specific_datasets: 특정 데이터셋만 처리 (예: ['TL_1', 'TL_3'])
                              None이면 모든 데이터셋 처리
            min_samples: 이 개수 미만인 클래스만 추가 (기본: 10)
            max_samples_per_class: 클래스당 최대 추가 개수 (None=무제한, 기본: None)
        """
        print("\n" + "="*60)
        print("AI Hub 데이터 통합 시작")
        print("="*60)

        # 현재 데이터 분석
        current_counts = self.analyze_current_data()

        # 사용 가능한 데이터셋 자동 탐색
        print("\n=== 데이터셋 탐색 ===")
        datasets = self.discover_available_datasets(specific_datasets)

        if not datasets:
            print("\n⚠️ 사용 가능한 데이터셋이 없습니다.")
            return

        print(f"\n총 {len(datasets)}개 데이터셋 발견")

        # 각 데이터셋 처리
        total_added = defaultdict(int)

        for dataset_name, paths in datasets.items():
            added_counts = self.process_aihub_dataset(
                dataset_name=dataset_name,
                source_anno_dir=paths['anno'],
                source_img_dir=paths['img'],
                current_class_counts=current_counts,
                min_samples=min_samples,
                max_samples_per_class=max_samples_per_class
            )

            # 누적
            for cls_id, count in added_counts.items():
                total_added[cls_id] += count
                current_counts[cls_id] = current_counts.get(cls_id, 0) + count

        # 최종 요약
        print("\n" + "="*60)
        print("통합 완료")
        print("="*60)

        if total_added:
            df = pd.DataFrame([
                {
                    'class_id': k,
                    'before': current_counts[k] - total_added[k],
                    'added': total_added[k],
                    'after': current_counts[k]
                }
                for k in sorted(total_added.keys())
            ])
            print("\n최종 결과:")
            print(df.to_string(index=False))
            print(f"\n총 추가 클래스: {len(total_added)}개")
            print(f"총 추가 이미지: {sum(total_added.values())}개")
        else:
            print("\n추가된 데이터 없음")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Hub 데이터 통합 (TS2/TL2 제외)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # API로 데이터 다운로드 (API 키 필요)
  python src/data/aihub_integration.py --download --api-key YOUR_KEY

  # 데이터셋 파일 목록 조회
  python src/data/aihub_integration.py --list-files --api-key YOUR_KEY

  # 로컬 데이터셋 통합 (data/added/ 폴더)
  python src/data/aihub_integration.py

  # 특정 데이터셋만 처리
  python src/data/aihub_integration.py --datasets TL_1 TL_3 TL_4

  # 현재 데이터 분석만 수행
  python src/data/aihub_integration.py --analyze-only

  # 클래스당 최대 개수 제한 (기본: 무제한)
  python src/data/aihub_integration.py --max-samples-per-class 100

참고:
  - TS2, TL2 데이터셋은 자동으로 제외됩니다
  - API 키는 .env.aihub 파일에서도 읽습니다
        """
    )
    parser.add_argument(
        '--project-root',
        type=str,
        default='.',
        help='프로젝트 루트 디렉토리 (기본: 현재 디렉토리)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='AI Hub API 키 (.env.aihub에서도 읽음)'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='API로 데이터 다운로드 (--api-key 필요)'
    )
    parser.add_argument(
        '--list-files',
        action='store_true',
        help='데이터셋 파일 목록 조회'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        type=str,
        help='처리할 데이터셋 지정 (예: TL_1 TL_3). 미지정시 모든 데이터셋 처리'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help='이 개수 미만인 클래스만 추가 (기본: 10)'
    )
    parser.add_argument(
        '--max-samples-per-class',
        type=int,
        default=None,
        help='클래스당 최대 추가 개수 (기본: 무제한)'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='현재 데이터 분석만 수행하고 종료'
    )

    args = parser.parse_args()

    # API 키 로드 (.env.aihub 또는 명령줄)
    api_key = args.api_key
    if not api_key:
        env_file = Path(args.project_root) / '.env.aihub'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('AIHUB_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        print(f"✅ API 키 로드: .env.aihub")
                        break

    # API 다운로드 모드
    if args.download or args.list_files:
        if not api_key:
            print("❌ API 키가 필요합니다. --api-key 옵션 또는 .env.aihub 파일을 사용하세요.")
            return

        downloader = AIHubDownloader(args.project_root, api_key)

        # aihubshell 설치
        if not downloader.install_aihubshell():
            print("❌ aihubshell 설치 실패")
            return

        # 파일 목록 조회
        if args.list_files:
            downloader.list_dataset_files()
            return

        # 다운로드
        if args.download:
            print("\n⚠️ 다운로드 기능은 수동으로 filekey를 지정해야 합니다.")
            print("먼저 --list-files로 filekey를 확인하세요.")
            downloader.list_dataset_files()
            return

    # 데이터 통합 모드
    integrator = AIHubDataIntegrator(args.project_root)

    # 분석만 수행
    if args.analyze_only:
        integrator.analyze_current_data()
        return

    # 전체 통합 프로세스
    integrator.integrate_all_data(
        specific_datasets=args.datasets,
        min_samples=args.min_samples,
        max_samples_per_class=args.max_samples_per_class
    )


if __name__ == '__main__':
    main()
