import Home from './pages/home';
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import './App.css';
import 'react-toastify/dist/ReactToastify.css';

function App() {
    return (
        <div className="App">
            <ToastContainer theme='colored' position='top-center'></ToastContainer>
            <BrowserRouter>
                <Routes>
                    <Route path = '/' element={<Home/>}></Route>
                </Routes>
            </BrowserRouter>
        </div>
    );
}

export default App;