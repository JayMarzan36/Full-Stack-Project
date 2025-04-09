import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import 'vite/modulepreload-polyfill'

import { createHashRouter, RouterProvider } from 'react-router'


//TODO go over canvas and notes to properly make more 'pages' for the single page application
//TODO make view for homepage
//TODO make view for making note
//TODO make view for viewing and deleting notes
//TODO make view for viewing graph
//TODO figure out how to make a view of the graph

const router = createHashRouter([
  {
    path: "/",
    element: <App/>,
  }
])

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>

    <RouterProvider router={router}/>
  
  </React.StrictMode>,
)
