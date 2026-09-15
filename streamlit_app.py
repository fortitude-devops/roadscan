"""
RoadScan — интерфейс демонстрации.

Запуск:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app import config, priority, privacy, quality, render, vision  # noqa: E402

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="RoadScan — приоритизация ремонта дорог", page_icon="🛣️", layout="wide")


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
@st.cache_data
def load_zones() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "zones.csv")
    df["context_flags"] = df["context_flags"].fillna("")
    return df


zones = load_zones()

if "queue" not in st.session_state:
    st.session_state.queue = []


# ---------------------------------------------------------------------------
# Сайдбар: контекст участка
# ---------------------------------------------------------------------------
st.sidebar.title("🛣️ RoadScan")
st.sidebar.caption("Выявление дефектов покрытия и приоритизация ремонта")

zone_row = None
mode = st.sidebar.radio("Контекст участка", ["Из справочника зон", "Задать вручную"], index=0)

if mode == "Из справочника зон":
    zone_label = st.sidebar.selectbox(
        "Участок",
        zones.index,
        format_func=lambda i: f"{zones.loc[i, 'zone_id']} — {zones.loc[i, 'zone_name']}",
    )
    zone_row = zones.loc[zone_label]
    zone_id = zone_row["zone_id"]
    road_type = zone_row["road_type"]
    traffic_level = zone_row["traffic_level"]
    flags = [f for f in str(zone_row["context_flags"]).split("|") if f]
    st.sidebar.info(
        f"**{zone_row['zone_name']}**\n\n"
        f"Тип: {config.ROAD_TYPES[road_type]}\n\n"
        f"Трафик: {config.TRAFFIC_LEVELS[traffic_level]}"
    )
else:
    zone_id = st.sidebar.text_input("ID участка", "ZONE-XX")
    road_type = st.sidebar.selectbox(
        "Тип дороги", list(config.ROAD_TYPES), format_func=lambda k: config.ROAD_TYPES[k], index=4
    )
    traffic_level = st.sidebar.selectbox(
        "Трафик", list(config.TRAFFIC_LEVELS), format_func=lambda k: config.TRAFFIC_LEVELS[k], index=1
    )
    flags = st.sidebar.multiselect(
        "Контекст", list(config.CONTEXT_BONUS), format_func=lambda k: config.CONTEXT_BONUS[k][0]
    )

weather = st.sidebar.selectbox(
    "Погода", list(config.WEATHER), format_func=lambda k: config.WEATHER[k]
)

st.sidebar.divider()
anonymize_on = st.sidebar.checkbox("Анонимизировать лица и номера", value=True)
use_cache = st.sidebar.checkbox("Использовать кэш", value=True,
                                help="Снимите, чтобы принудительно перезапросить модель")

if not os.getenv("GEMINI_API_KEY") and os.getenv("OFFLINE_MODE") != "1":
    st.sidebar.error("Не найден GEMINI_API_KEY — заполните файл .env")


# ---------------------------------------------------------------------------
# Основная область
# ---------------------------------------------------------------------------
tab_analyze, tab_queue, tab_rules = st.tabs(["🔍 Анализ фото", "📋 Очередь ремонта", "⚙️ Логика приоритета"])

with tab_analyze:
    uploaded = st.file_uploader(
        "Фотография дорожного покрытия", type=["jpg", "jpeg", "png", "webp"],
        help="Фото жителя, служебного объезда или кадр из видео",
    )

    if uploaded is not None:
        raw = np.frombuffer(uploaded.getvalue(), np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Не удалось прочитать изображение")
            st.stop()

        privacy_found: list[str] = []
        if anonymize_on:
            image, privacy_found = privacy.anonymize(image)

        qrep = quality.assess(image)

        with st.spinner("Анализ снимка…"):
            try:
                vres = vision.analyze(
                    image,
                    {
                        "zone_id": zone_id,
                        "road_type_label": config.ROAD_TYPES.get(road_type, road_type),
                        "weather_label": config.WEATHER.get(weather, weather),
                    },
                    use_cache=use_cache,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ошибка обращения к модели: {exc}")
                st.stop()

        det = vres.detection
        pres = priority.compute(
            defect_type=det.defect_type,
            severity=det.severity,
            model_confidence=det.confidence,
            quality_score=qrep.score,
            road_type=road_type,
            traffic_level=traffic_level,
            weather=weather,
            context_flags=flags,
        )

        col_img, col_res = st.columns([1.15, 1])

        with col_img:
            caption = f"{pres.level.upper()} {pres.score:.1f}"  # только латиница: OpenCV не рисует кириллицу
            shown = render.draw_bbox(image, det.bbox, pres.level, caption) if det.bbox else image
            st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), width="stretch")
            st.caption(
                f"Разрешение {qrep.width}×{qrep.height} · резкость {qrep.blur_var:.0f} · "
                f"яркость {qrep.brightness:.0f} · {'из кэша' if vres.from_cache else vres.model}"
            )

        with col_res:
            if det.defect_type == "none":
                st.success("### Дефекта не обнаружено")
            else:
                st.markdown(
                    f"<div style='background:{pres.color};color:#fff;padding:14px 18px;"
                    f"border-radius:10px;font-size:20px;font-weight:600'>"
                    f"{config.DEFECT_TYPES.get(det.defect_type, det.defect_type)} · "
                    f"приоритет: {pres.label}</div>",
                    unsafe_allow_html=True,
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("Балл приоритета", f"{pres.score:.2f}")
            m2.metric("Уверенность", f"{pres.final_confidence:.0%}")
            m3.metric("Срок реакции", f"{pres.sla_days} дн." if pres.sla_days else "—")

            st.write(f"**Локализация:** {det.location_text}")
            st.write(det.description)

            if det.evidence:
                with st.expander("Признаки, на которых основан вывод", expanded=True):
                    for e in det.evidence:
                        st.write(f"• {e}")

            if det.secondary_defects:
                st.caption("Дополнительно замечено: " + ", ".join(det.secondary_defects))

            if pres.needs_manual_review:
                st.warning(f"⚠️ **Требуется ручная проверка.** {pres.review_reason}")
            if qrep.issues:
                st.info("Качество снимка: " + "; ".join(qrep.issues))
            if privacy_found:
                st.caption("🔒 Анонимизировано: " + ", ".join(sorted(set(privacy_found))))

        # --- Разбор приоритета по факторам ---------------------------------
        if det.defect_type != "none":
            st.subheader("Почему такой приоритет")
            fdf = pd.DataFrame(
                [{"Фактор": f.explanation, "Вклад, баллы": round(f.contribution, 2)} for f in pres.factors]
            )
            c1, c2 = st.columns([1.4, 1])
            c1.dataframe(fdf, width="stretch", hide_index=True)
            c2.bar_chart(fdf.set_index("Фактор")["Вклад, баллы"], horizontal=True)
            st.caption(
                f"Итог: {pres.score:.2f} балла. Пороги: <{config.PRIORITY_THRESHOLDS['medium']} низкий, "
                f"{config.PRIORITY_THRESHOLDS['medium']}–{config.PRIORITY_THRESHOLDS['high']} средний, "
                f"≥{config.PRIORITY_THRESHOLDS['high']} высокий."
            )

            # --- Что если контекст другой? ---------------------------------
            with st.expander("🔄 Сравнить: тот же дефект в другом контексте"):
                alt = st.selectbox(
                    "Перенести дефект на участок типа", list(config.ROAD_TYPES),
                    format_func=lambda k: config.ROAD_TYPES[k],
                    index=list(config.ROAD_TYPES).index("yard"),
                    key="alt_road",
                )
                alt_traffic = st.select_slider(
                    "Трафик", list(config.TRAFFIC_LEVELS), value="low",
                    format_func=lambda k: config.TRAFFIC_LEVELS[k], key="alt_traffic",
                )
                alt_res = priority.compute(
                    det.defect_type, det.severity, det.confidence, qrep.score,
                    alt, alt_traffic, weather, [],
                )
                a, b = st.columns(2)
                a.metric(f"Сейчас: {config.ROAD_TYPES[road_type]}", pres.label, f"{pres.score:.2f} балла")
                b.metric(f"Если: {config.ROAD_TYPES[alt]}", alt_res.label, f"{alt_res.score:.2f} балла")
                st.caption("Один и тот же дефект, разный контекст — разный приоритет и срок реакции.")

        if st.button("➕ Добавить в очередь ремонта", type="primary"):
            st.session_state.queue.append({
                "image_id": f"ROAD-{len(st.session_state.queue) + 1:03d}",
                "zone_id": zone_id,
                "Участок": zone_row["zone_name"] if zone_row is not None else "—",
                "Дефект": config.DEFECT_TYPES.get(det.defect_type, det.defect_type),
                "Приоритет": pres.label,
                "Балл": pres.score,
                "Уверенность": round(pres.final_confidence, 2),
                "Срок, дн.": pres.sla_days,
                "Ручная проверка": "да" if pres.needs_manual_review else "нет",
                "lat": float(zone_row["lat"]) if zone_row is not None else None,
                "lon": float(zone_row["lon"]) if zone_row is not None else None,
            })
            st.success("Заявка добавлена в очередь")

with tab_queue:
    st.subheader("Очередь ремонта")
    if not st.session_state.queue:
        st.info("Очередь пуста. Проанализируйте фото и нажмите «Добавить в очередь ремонта».")
    else:
        qdf = pd.DataFrame(st.session_state.queue).sort_values("Балл", ascending=False)
        k1, k2, k3 = st.columns(3)
        k1.metric("Всего заявок", len(qdf))
        k2.metric("Высокий приоритет", int((qdf["Приоритет"] == "Высокий").sum()))
        k3.metric("На ручной проверке", int((qdf["Ручная проверка"] == "да").sum()))

        st.dataframe(qdf.drop(columns=["lat", "lon"]), width="stretch", hide_index=True)

        geo = qdf.dropna(subset=["lat", "lon"])
        if not geo.empty:
            st.map(geo[["lat", "lon"]], size=60)

        st.download_button(
            "⬇️ Выгрузить наряд-задание (CSV)",
            qdf.drop(columns=["lat", "lon"]).to_csv(index=False).encode("utf-8-sig"),
            file_name="repair_queue.csv",
            mime="text/csv",
        )
        if st.button("Очистить очередь"):
            st.session_state.queue = []
            st.rerun()

with tab_rules:
    st.subheader("Как считается приоритет")
    st.code(
        "score = severity × 2\n"
        "      × вес_типа_дефекта\n"
        "      × вес_типа_дороги\n"
        "      × вес_трафика\n"
        "      + бонус_погоды\n"
        "      + бонусы_контекста (школа, переход, мост, поворот…)\n"
        "\n"
        "если итоговая уверенность < 0.55 → приоритет ограничен «средним»\n"
        "                                  и заявка уходит на ручную проверку",
        language="text",
    )
    c1, c2 = st.columns(2)
    c1.markdown("**Вес типа дороги**")
    c1.dataframe(
        pd.DataFrame(
            [{"Тип дороги": config.ROAD_TYPES[k], "Коэффициент": v} for k, v in config.ROAD_TYPE_WEIGHT.items()]
        ),
        hide_index=True, width="stretch",
    )
    c2.markdown("**Контекстные надбавки**")
    c2.dataframe(
        pd.DataFrame([{"Фактор": v[0], "Баллы": v[1]} for v in config.CONTEXT_BONUS.values()]),
        hide_index=True, width="stretch",
    )
    st.caption(
        "Веса заданы в app/config.py и меняются без правки кода. "
        "Приоритет не назначает санкций подрядчику — это рекомендация для планирования работ."
    )
