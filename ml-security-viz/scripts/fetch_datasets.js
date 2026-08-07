const fs = require('fs');
const https = require('https');
const path = require('path');

const fetchCSV = (url) => {
  return new Promise((resolve, reject) => {
    let data = '';
    https.get(url, { headers: { 'User-Agent': 'Node.js' } }, (res) => {
      // Follow redirects
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return resolve(fetchCSV(res.headers.location));
      }
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve(data));
    }).on('error', reject);
  });
};

const normalize = (x, y) => {
  const x_min = Math.min(...x);
  let x_max = Math.max(...x);
  if (x_min === x_max) x_max += 1e-5;
  const nx = x.map(v => ((v - x_min) / (x_max - x_min)) * 4 - 2);

  const y_min = Math.min(...y);
  let y_max = Math.max(...y);
  if (y_min === y_max) y_max += 1e-5;
  const ny = y.map(v => ((v - y_min) / (y_max - y_min)) * 4 - 2);

  return { nx, ny };
};

const processDataset = async (name, url, xCol, yCol) => {
  try {
    const csv = await fetchCSV(url);
    const lines = csv.split('\n').map(l => l.split(','));
    const headers = lines[0].map(h => h.trim().replace(/"/g, ''));
    
    // Find approximate matches if exact is not found
    let xIdx = headers.findIndex(h => h.toLowerCase().includes(xCol.toLowerCase()));
    let yIdx = headers.findIndex(h => h.toLowerCase().includes(yCol.toLowerCase()));
    
    if (xIdx === -1) xIdx = headers.indexOf(xCol);
    if (yIdx === -1) yIdx = headers.indexOf(yCol);
    
    if (xIdx === -1 || yIdx === -1) {
      console.error(`[${name}] Missing columns: ${xCol} or ${yCol}. Found: ${headers}`);
      return;
    }

    let x = [], y = [];
    for (let i = 1; i < lines.length; i++) {
      if (lines[i].length > Math.max(xIdx, yIdx)) {
        const w = parseFloat(lines[i][xIdx]);
        const d = parseFloat(lines[i][yIdx]);
        if (!isNaN(w) && !isNaN(d)) {
          x.push(w);
          y.push(d);
        }
      }
    }
    
    const sample = [];
    for (let i = 0; i < x.length; i++) sample.push([x[i], y[i]]);
    sample.sort(() => 0.5 - Math.random());
    const subset = sample.slice(0, 150);
    
    const { nx, ny } = normalize(subset.map(s => s[0]), subset.map(s => s[1]));
    const data = { X: nx.map(v => [v]), y: ny };
    fs.writeFileSync(path.join(__dirname, '..', 'src', 'engine', `${name}.json`), JSON.stringify(data));
    console.log(`[${name}] saved ${subset.length} points.`);
  } catch (err) {
    console.error(`[${name}] Error:`, err.message);
  }
};

const run = async () => {
  // Medical/Health Care (Proxy for Warfarin)
  await processDataset('warfarin', 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv', 'bmi', 'charges');
  
  // Real Estate (House Pricing)
  await processDataset('housePricing', 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv', 'median_income', 'median_house_value');
  
  // Finance (Proxy for Lending Club)
  await processDataset('lendingClub', 'https://raw.githubusercontent.com/gastonstat/CreditScoring/master/CleanCreditScoring.csv', 'Age', 'Income'); 
};

run();
