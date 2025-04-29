/// <reference types="react-scripts" />

// Add React JSX definitions
declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
}
