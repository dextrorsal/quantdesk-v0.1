module.exports = {
  'src/**/*.{ts,tsx}': [
    'eslint --fix',
    'prettier --write',
    'git add'
  ],
  'src/**/*.{js,jsx}': [
    'eslint --fix',
    'prettier --write',
    'git add'
  ],
  '*.{json,md,yml,yaml}': [
    'prettier --write',
    'git add'
  ]
};
