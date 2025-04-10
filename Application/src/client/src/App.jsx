import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

import { Outlet } from 'react-router';
import { Link } from 'react-router';

function App() {
  const [count, setCount] = useState(0)

  async function logout() {
    const res = await fetch("/registration/logout/", {
      credentials: "same-origin", // TODO include cookies!
    });

    if (res.ok) {
      // TODO navigate away from the single page app!
      window.location = "/registration/sign_in/";
    } else {
      // TODO handle logout failed!
    }
  }

  return (
    <>
      <nav className="Navigation">

        <div>
            <Link>New Note</Link>
        </div>
        
        <div>
          <Link>View Notes</Link>
        </div>

        <div>
          <button onClick={logout}>Logout</button>
        </div>

      </nav>


      <main>
        <Outlet>{ }</Outlet>
      </main>

    </>
  )
}

export default App;