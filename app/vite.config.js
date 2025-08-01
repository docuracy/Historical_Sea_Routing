// vite.config.js
export default {
  base: '/Historical_Sea_Routing/', // Required for GitHub Pages
  publicDir: 'public',
  build: {
    outDir: '../docs', // build into the root-level 'docs/' folder
    emptyOutDir: false
  },
  // server: {
  //   configureServer(server) {
  //     server.middlewares.use((req, res, next) => {
  //       if (req.url.endsWith('.pmtiles')) {
  //         res.setHeader('Content-Type', 'application/octet-stream');
  //         res.setHeader('Content-Encoding', 'identity'); // Crucial: disable compression
  //       }
  //       next();
  //     });
  //   }
  // }
};