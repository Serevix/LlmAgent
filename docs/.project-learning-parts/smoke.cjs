const { chromium } = require('C:/Users/19344/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1360, height: 920 } });
  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  await page.goto('http://127.0.0.1:8420/index.html', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('.section-block.active', { timeout: 30000 });
  const count = await page.locator('.section-block').count();
  if (count < 30) throw new Error('Missing sections: ' + count);
  await page.screenshot({ path: path.join(__dirname, 'desktop.png'), fullPage: true, animations: 'disabled' });

  const info = await page.evaluate(() => ({
    sections: allSections.length,
    practice: allSections.filter(x => x.chapterTitle === '实战演练').length,
    mermaidAvailable: typeof mermaid !== 'undefined',
    codeIndex: allSections.findIndex(x => x.html.includes('callchain-code-toggle')),
    questionIndex: allSections.findIndex(x => x.html.includes('question-card')),
    diagramIndex: allSections.findIndex(x => x.html.includes('class="mermaid"')),
  }));
  await page.evaluate(idx => navigateToSection(idx), info.codeIndex);
  const codeToggle = page.locator('.section-block.active .callchain-code-toggle').first();
  await codeToggle.waitFor({ state: 'visible' });
  await codeToggle.click();
  if (await codeToggle.getAttribute('aria-expanded') !== 'true') throw new Error('Code expansion failed');
  await page.screenshot({ path: path.join(__dirname, 'callchain.png'), fullPage: true, animations: 'disabled' });

  await page.locator('#searchInput').fill('摘要');
  await page.locator('.search-result-item').first().waitFor({ state: 'visible' });
  await page.locator('.search-result-item').first().click();
  await page.waitForFunction(() => !document.getElementById('searchResults').classList.contains('open'));

  await page.evaluate(idx => navigateToSection(idx), info.questionIndex);
  const question = page.locator('.section-block.active .question-header').first();
  await question.waitFor({ state: 'visible' });
  await question.click();
  if (await question.getAttribute('aria-expanded') !== 'true') throw new Error('Question expansion failed');

  if (!info.mermaidAvailable) throw new Error('Mermaid unavailable');
  const diagramErrors = await page.evaluate(async () => {
    const failed = [];
    for (const ch of docData.chapters) for (const sec of ch.sections) for (const block of sec.blocks) {
      if (block.type === 'diagram') {
        try { await mermaid.parse(block.diagram.content); }
        catch (err) { failed.push(sec.title + ': ' + err.message); }
      }
    }
    return failed;
  });
  if (diagramErrors.length) throw new Error('Diagram parsing failed: ' + JSON.stringify(diagramErrors));
  await page.evaluate(idx => navigateToSection(idx), info.diagramIndex);
  await page.locator('.section-block.active .mermaid svg').first().waitFor({ state: 'visible', timeout: 20000 });
  await page.locator('.section-block.active .diagram-view-btn').first().click();
  if (!await page.locator('#diagramDialog').evaluate(el => el.open)) throw new Error('Diagram enlargement failed');
  await page.locator('#diagramDialog .section-nav-btn').click();
  await page.screenshot({ path: path.join(__dirname, 'architecture.png'), fullPage: true, animations: 'disabled' });

  await page.locator('#themeToggle').click();
  if (await page.locator('html').getAttribute('data-theme') !== 'dark') throw new Error('Theme toggle failed');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => navigateToSection(0));
  await page.waitForFunction(() => currentSectionIdx === 0);
  await page.locator('#hamburgerBtn').click();
  if (!await page.locator('#sidebar').evaluate(el => el.classList.contains('open'))) throw new Error('Mobile menu failed');
  await page.locator('#hamburgerBtn').click();
  await page.screenshot({ path: path.join(__dirname, 'mobile.png'), fullPage: true, animations: 'disabled' });
  const clippedButtons = await page.locator('#sectionNav button').evaluateAll(els => els.filter(el => {
    const box = el.getBoundingClientRect();
    return box.left < 0 || box.right > innerWidth;
  }).length);
  if (clippedButtons) throw new Error('Mobile navigation buttons clipped');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
  if (overflow) throw new Error('Mobile page overflows horizontally');

  const offline = await browser.newPage();
  await offline.goto(pathToFileURL(path.resolve(__dirname, '../index.html')).href, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await offline.waitForSelector('.section-block.active', { timeout: 20000 });
  if (await offline.locator('.section-block').count() !== count) throw new Error('File-open fallback mismatch');
  if (errors.length) throw new Error('Browser errors: ' + JSON.stringify(errors));
  console.log(JSON.stringify({ ...info, codeExpansion: true, search: true, interview: true,
    diagramParsing: true, theme: true, mobile: true, fileOpenFallback: true, browserErrors: errors }, null, 2));
  await browser.close();
})().catch(err => { console.error(err); process.exit(1); });
