(() => {
  const form = document.getElementById("analyze-form");
  const submitBtn = document.getElementById("submit-btn");
  const statusEl = document.getElementById("status");
  const errorEl = document.getElementById("error");
  const summaryEl = document.getElementById("summary");
  const reportSection = document.getElementById("report-section");
  const reportBody = document.getElementById("report-body");
  const reportMeta = document.getElementById("report-meta");
  const metadataJson = document.getElementById("metadata-json");

  function show(el, showIt) {
    el.classList.toggle("hidden", !showIt);
  }

  function resetOutput() {
    show(errorEl, false);
    show(summaryEl, false);
    show(reportSection, false);
    reportBody.innerHTML = "";
    reportMeta.textContent = "";
    metadataJson.textContent = "";
  }

  function setLoading(loading) {
    submitBtn.disabled = loading;
    show(statusEl, loading);
    statusEl.textContent = loading
      ? "Выполняется пайплайн: извлечение фактов, поиск соседей, генерация отчёта…"
      : "";
    statusEl.classList.toggle("loading", loading);
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resetOutput();

    const fileInput = document.getElementById("file");
    const reportNameInput = document.getElementById("report_name");

    if (!fileInput.files?.length) {
      show(errorEl, true);
      errorEl.textContent = "Выберите файл.";
      return;
    }

    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    const name = reportNameInput.value.trim();
    if (name) {
      fd.append("report_name", name);
    }

    setLoading(true);

    try {
      const res = await fetch("/contradiction/detect_contradictions", {
        method: "POST",
        body: fd,
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        const detail = data.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg || JSON.stringify(d)).join("\n")
              : detail != null
                ? JSON.stringify(detail, null, 2)
                : `HTTP ${res.status}`;
        throw new Error(msg);
      }

      if (data.status !== "success") {
        throw new Error(data.error || "Пайплайн завершился с ошибкой.");
      }

      document.getElementById("stat-chunks").textContent = String(data.chunks_count ?? "—");
      document.getElementById("stat-facts").textContent = String(data.facts_count ?? "—");
      document.getElementById("stat-contradictions").textContent = String(
        data.contradictions_count ?? "—",
      );

      if (data.metadata && typeof data.metadata === "object") {
        metadataJson.textContent = JSON.stringify(data.metadata, null, 2);
      } else {
        metadataJson.textContent = "—";
      }

      show(summaryEl, true);

      const report = data.report;
      const md =
        report && typeof report.report === "string"
          ? report.report
          : typeof report === "string"
            ? report
            : "";

      if (!md.trim()) {
        throw new Error("В ответе нет текста отчёта (поле report.report).");
      }

      const html = marked.parse(md, { gfm: true, breaks: false });
      reportBody.innerHTML = DOMPurify.sanitize(html);

      const parts = [];
      if (report && report.model) {
        parts.push(`Модель: ${report.model}`);
      }
      if (report && report.timestamp) {
        parts.push(report.timestamp);
      }
      if (report && report.file_path) {
        parts.push(`Файл: ${report.file_path}`);
      }
      reportMeta.textContent = parts.join(" · ");

      show(reportSection, true);
      reportSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      show(errorEl, true);
      errorEl.textContent = err instanceof Error ? err.message : String(err);
    } finally {
      setLoading(false);
    }
  });
})();
