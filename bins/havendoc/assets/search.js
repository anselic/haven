(() => {
  const trigger = document.querySelector('.search-trigger');
  const dialog = document.querySelector('#search-dialog');
  const input = document.querySelector('#search-input');
  const results = document.querySelector('.search-results');
  const status = document.querySelector('.search-status');
  const close = document.querySelector('.search-close');
  const entries = window.havenSearchEntries || [];
  if (!trigger || !dialog || !input || !results || !status || !close) return;

  const render = () => {
    const query = input.value.trim().toLocaleLowerCase();
    results.replaceChildren();
    if (!query) {
      status.textContent = 'Type to search the documentation.';
      return;
    }

    const matches = entries
      .filter(([label]) => label.toLocaleLowerCase().includes(query))
      .sort(([left], [right]) => {
        const leftStarts = left.toLocaleLowerCase().startsWith(query);
        const rightStarts = right.toLocaleLowerCase().startsWith(query);
        return Number(rightStarts) - Number(leftStarts) || left.localeCompare(right);
      })
      .slice(0, 50);
    status.textContent = matches.length
      ? `${matches.length} result${matches.length === 1 ? '' : 's'}`
      : 'No results found.';

    for (const [label, kind, path] of matches) {
      const item = document.createElement('li');
      const link = document.createElement('a');
      const name = document.createElement('span');
      const category = document.createElement('span');
      link.href = trigger.dataset.root + path;
      name.textContent = label;
      category.className = 'search-result-kind';
      category.textContent = kind;
      link.append(name, category);
      item.append(link);
      results.append(item);
    }
  };

  const open = () => {
    dialog.showModal();
    input.value = '';
    render();
    input.focus();
  };

  trigger.addEventListener('click', open);
  close.addEventListener('click', () => dialog.close());
  input.addEventListener('input', render);
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      const first = results.querySelector('a');
      if (first) first.click();
    }
  });
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  });
  document.addEventListener('keydown', (event) => {
    const target = event.target;
    const editing = target instanceof HTMLElement &&
      (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
    if (!dialog.open && !editing && (event.key === '/' || ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k'))) {
      event.preventDefault();
      open();
    }
  });
})();
