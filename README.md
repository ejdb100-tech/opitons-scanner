# 미국 개별주 옵션 스캐너 (Max Pain / Call Wall / Put Wall)

여러 미국 종목의 옵션 체인을 받아 **Max Pain · 콜월 · 풋월**을 월물/주물로 비교하는 Streamlit 앱.

## 기능
- 멀티 종목 스캐너 (티커 여러 개 입력 → 월물·주물 핵심 레벨 표)
- 종목 상세: 월물 / 주물 OI 차트 나란히 비교
- 감마 프로파일(GEX, 딜러 포지션 가정 토글)

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 데이터
- Yahoo Finance (`yfinance`). OI는 보통 전일 종가 기준 1일 지연.
- 보조 지표일 뿐이며 투자 신호가 아닙니다.

## 배포 (Streamlit Community Cloud)
1. 이 저장소를 GitHub에 push
2. https://share.streamlit.io 에서 GitHub 로그인
3. "Create app" → 저장소 / 브랜치 / `app.py` 지정 → Deploy
