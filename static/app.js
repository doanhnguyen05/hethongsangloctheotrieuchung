document.addEventListener("DOMContentLoaded", () => {
  const ua = navigator.userAgent || "";
  const isSafari = /^((?!chrome|android|crios|edg|opr|coc_coc_browser).)*safari/i.test(ua);
  const isChromiumFamily =
    /chrome|chromium|crios|edg|opr|coc_coc_browser/i.test(ua) && !isSafari;

  if (isChromiumFamily) {
    document.body.classList.add("chromium-optimized");
  }

  document.querySelectorAll(".symptom-card").forEach((card) => {
    const input = card.querySelector('input[type="checkbox"]');
    if (!input) {
      return;
    }

    const syncState = () => {
      card.classList.toggle("is-checked", input.checked);
    };

    syncState();
    input.addEventListener("change", syncState, { passive: true });
  });

  const symptomSearchInput = document.querySelector("#symptom-search");
  const symptomGroups = Array.from(document.querySelectorAll("[data-symptom-group]"));
  const searchSummary = document.querySelector("#symptom-search-summary");
  const emptyState = document.querySelector("#symptom-search-empty");

  if (!symptomSearchInput || !symptomGroups.length || !searchSummary || !emptyState) {
    return;
  }

  const normalizeText = (value) =>
    (value || "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .trim();

  const totalSymptoms = symptomGroups.reduce((sum, group) => {
    return sum + group.querySelectorAll("[data-symptom-item]").length;
  }, 0);

  const updateSymptomSearch = () => {
    const rawKeyword = symptomSearchInput.value || "";
    const normalizedKeyword = normalizeText(rawKeyword);
    let visibleSymptoms = 0;
    let visibleGroups = 0;

    symptomGroups.forEach((group) => {
      const items = Array.from(group.querySelectorAll("[data-symptom-item]"));
      let groupVisibleCount = 0;

      items.forEach((item) => {
        const symptomName = normalizeText(item.dataset.symptomName || item.textContent);
        const isMatch = !normalizedKeyword || symptomName.includes(normalizedKeyword);
        item.hidden = !isMatch;
        if (isMatch) {
          groupVisibleCount += 1;
          visibleSymptoms += 1;
        }
      });

      group.hidden = groupVisibleCount === 0;
      if (groupVisibleCount > 0) {
        visibleGroups += 1;
      }

      const countNode = group.querySelector("[data-group-count]");
      if (countNode) {
        countNode.textContent = `${groupVisibleCount}/${items.length} triệu chứng`;
      }
    });

    if (!normalizedKeyword) {
      searchSummary.textContent = `Hiển thị toàn bộ ${totalSymptoms} triệu chứng ở ${symptomGroups.length} nhóm`;
      emptyState.hidden = true;
      return;
    }

    searchSummary.textContent = `Tìm thấy ${visibleSymptoms} triệu chứng trong ${visibleGroups} nhóm cho từ khóa "${rawKeyword.trim()}"`;
    emptyState.hidden = visibleSymptoms !== 0;
  };

  symptomSearchInput.addEventListener("input", updateSymptomSearch, { passive: true });
  symptomSearchInput.addEventListener("search", updateSymptomSearch, { passive: true });
  updateSymptomSearch();
});
