// vite.config.js
export default {
  base: '/Historical_Sea_Routing/', // Required for GitHub Pages
  publicDir: 'public',
  build: {
    outDir: '../docs', // build into the root-level 'docs/' folder
    emptyOutDir: false
  }
};