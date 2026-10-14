// Dependency-free DOM unit tests: node --test tests/test_library_ui.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function setup() {
  class Element {
    constructor() { this.children = []; this.events = {}; this.value = ''; }
    addEventListener(name, handler) { this.events[name] = handler; }
    appendChild(child) { this.children.push(child); }
    replaceChildren() { this.children = []; }
    set innerHTML(_) { this.children = []; }
    setAttribute(key, value) { this[key] = value; }
  }
  const nodes = new Map();
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, new Element());
    return nodes.get(id);
  };
  const sent = [], notices = [];
  const context = vm.createContext({
    document: { getElementById: node, createElement: () => new Element(),
      querySelector: () => ({value: 'balanced'}) },
    window: {location: {}}, sessionStorage: {setItem() {}, removeItem() {}, getItem() {}},
    setTimeout, clearTimeout,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8'), context);
  context.api = () => new Promise(() => {}); // Leave initial settings load pending.
  context.post = async (url, body) => { sent.push({url, body}); return {jobs: [{job_id: 1}, {job_id: 2}]}; };
  context.toast = message => notices.push(message);
  context.readCategoryChips = () => ['profanity'];
  context.readActionGrid = () => ({profanity: 'mute'});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/library.js'), 'utf8'), context);
  return {context, node, sent, notices, run: code => vm.runInContext(code, context)};
}

test('select across folders, deduplicate, remove and clear', () => {
  const s = setup();
  s.run(`renderListing({path:'/one', dirs:[], files:[{path:'/one/a.mp4',name:'a'}]});
    addFiles(currentListing.files);
    renderListing({path:'/two', dirs:[], files:[{path:'/two/b.mp4',name:'b'}]});
    addFiles(currentListing.files); addFiles(currentListing.files);`);
  assert.equal(s.node('selection-count').textContent, '2 selected');
  s.node('selected-files').children[0].children[1].events.click();
  assert.equal(s.run('picked.size'), 1);
  s.node('cancel-pick').events.click();
  assert.equal(s.run('picked.size'), 0);
  assert.equal(s.node('scan-panel').hidden, true);
});

test('folder inclusion is selectable and queue waits for pending expansion', async () => {
  const s = setup();
  let complete;
  s.context.post = (url, body) => {
    s.sent.push({url, body}); return new Promise(resolve => {complete = resolve;});
  };
  s.run(`renderListing({path:'/NAS', dirs:[], files:[]});
    addFiles([{path:'/NAS/a.mp4',name:'a'}]);`);
  s.node('include-subfolders').checked = true;
  const pending = s.run('addFolder()');
  await s.run('startScan()');
  assert.equal(s.sent.length, 1);
  assert.equal(s.sent[0].body.recursive, true);
  complete({files: [{path:'/NAS/a.mp4',name:'a'}, {path:'/NAS/sub/b.mp4',name:'b'}]});
  await pending;
  assert.equal(s.run('picked.size'), 2);
  assert.equal(s.node('start-scan').disabled, false);
});

test('batch submit uses shared settings and navigates to Jobs', async () => {
  const s = setup();
  s.run(`addFiles([{path:'/a.mp4',name:'a'},{path:'/b.mp4',name:'b'}])`);
  s.node('word-end-padding-ms').value = '300';
  s.node('auto-render').checked = true;
  await s.run('startScan()');
  assert.equal(s.sent[0].url, '/api/scan');
  assert.deepEqual(Array.from(s.sent[0].body.paths), ['/a.mp4', '/b.mp4']);
  assert.equal(s.sent[0].body.word_end_padding_ms, 300);
  assert.equal(s.sent[0].body.auto_render, true);
  assert.equal(s.context.window.location.href, '/jobs');
});

test('oversized addition leaves existing selection intact', () => {
  const s = setup();
  s.run(`addFiles([{path:'/a.mp4',name:'a'}]);
    addFiles(Array.from({length:501}, (_,i)=>({path:'/file'+i+'.mp4',name:String(i)})));`);
  assert.equal(s.run('picked.size'), 1);
  assert.match(s.notices[0], /Nothing was added/);
});

test('clearing during folder expansion does not unexpectedly restore selection', async () => {
  const s = setup();
  let complete;
  s.context.post = () => new Promise(resolve => {complete = resolve;});
  s.run(`renderListing({path:'/NAS', dirs:[], files:[]});`);
  const pending = s.run('addFolder()');
  s.node('cancel-pick').events.click();
  complete({files: [{path:'/NAS/a.mp4',name:'a'}]});
  await pending;
  assert.equal(s.run('picked.size'), 0);
});
