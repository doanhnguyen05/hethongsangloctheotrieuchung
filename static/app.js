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
});
