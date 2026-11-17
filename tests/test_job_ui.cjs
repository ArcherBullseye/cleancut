const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function setup() {
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.events = {}; }
    addEventListener(name, handler) { this.events[name] = handler; }
    setAttribute(key, value) { this[key] = value; }
    appendChild(child) { child.parent = this; this.children.push(child); }
    querySelector(selector) {
      if (selector === `.${this.className}` || selector === this.tag) return this;
      return this.children.map(child => child.querySelector(selector)).find(Boolean) || null;
    }
    remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  }
  const root = new Element('section');
  root.dataset = {jobId: '12', blurPreviewSeconds: '8'};
  const card = new Element('div');
  const body = new Element('div');
  body.className = 'body';
  card.appendChild(body);
  const context = vm.createContext({
    document: {getElementById: id => id === 'job-root' ? root : new Element('div'),
      createElement: tag => new Element(tag), querySelectorAll: () => []},
    card,
    decision: {action: 'blur', index: 3, start: 1.234567, end: 18.456789},
  });
  const run = code => vm.runInContext(code, context);
  const load = file => run(fs.readFileSync(path.join(__dirname, '../static', file), 'utf8'));
  load('app.js');
  context.api = () => new Promise(() => {});
  load('job.js');
  return {context, run, card};
}

test('blur preview switches modes and pages in one player', () => {
  const {run, card} = setup();
  const click = text => card.querySelector('.clip-preview').children[0].children
    .find(button => button.textContent === text).events.click();
  run('showClip(card, decision)');
  assert.equal(card.querySelector('video').src, '/api/job/12/blur/3?page=0');
  click('Next segment');
  assert.equal(card.querySelector('video').src, '/api/job/12/blur/3?page=1');
  click('Original preview');
  const source = new URL(card.querySelector('video').src, 'http://localhost');
  assert.equal(source.pathname, '/api/job/12/clip');
  assert.equal(Number(source.searchParams.get('start')), 1.234567 + 8);
  assert.equal(Number(source.searchParams.get('end')), 1.234567 + 16);
  click('Blurred preview');
  assert.equal(card.querySelector('video').src, '/api/job/12/blur/3?page=1');
  assert.equal(card.querySelector('.body').children.length, 1);
  click('Close preview');
  assert.equal(card.querySelector('video'), null);
});

test('trim prompts preserve exact source timestamps', async () => {
  const {run, context} = setup();
  let saved;
  context.prompt = (_message, value) => value;
  context.edit = (_index, changes) => { saved = changes; };
  await run('trim(decision)');
  assert.equal(saved.start, '1.234567');
  assert.equal(saved.end, '18.456789');
});
