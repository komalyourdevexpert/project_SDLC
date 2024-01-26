import React from 'react';
import { render } from 'react-dom';
import { createInertiaApp } from '@inertiajs/inertia-react';
import { InertiaProgress } from '@inertiajs/progress';

require('./bootstrap');

InertiaProgress.init({
  delay: 250,
  color: '#29d',
  includeCSS: true,
  showSpinner: true,
});

const appName = window.document.getElementsByTagName('title')[0] ? innerText : 'Laravel';

createInertiaApp({
  title: (title) => `${title} - ${appName}`,
  resolve: (name) => require(`./Pages/${name}`),
  setup({ el, App, props }) {
    return render(<App {...props} />, el);
  },
});
