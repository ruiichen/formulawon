import { useEffect, useState } from "react";
import { toast } from "react-toastify";

const Home = () => {
    const [stage, setStage] = useState(1);
    const [season, setSeason] = useState(0);
    const [round, setRound] = useState(0);
    const [prediction, setPrediction] = useState(1);
    const [data, setData] = useState(null);

    const search = (e) => {
        e.preventDefault();
        const cur_year = new Date().getFullYear();
        if(season < 2003 || season > cur_year) {
            toast.error("The season is invalid!");
        } else {
            fetch(`https://ergast.com/api/f1/${season}/${round}/qualifying.json`, {
                method: 'GET', headers: {}, referrerPolicy: "unsafe-url"
            }).then((res) => {
                if (!res.ok) {
                    throw Error('ERGAST API returned error.');
                }
                return res.json();
            }).then((resp) => {
                if (resp.MRData.RaceTable.Races === undefined || resp.MRData.RaceTable.Races.length == 0) {
                    // array does not exist or is empty

                    toast.error("ERROR: The round is invalid.");
                } else {
                    toast.success("yap");
                    setData(resp.MRData.RaceTable.Races)
                    console.log(resp);
                    setStage(2)
                }
            }).catch((err) => {
                toast.error('Failed: ' + err.message);
            });
        }
    }


    const predict = (e) => {
        e.preventDefault();
        console.log(prediction);
        var predict_url = '';
        switch(parseInt(prediction)) {
            case 1:
                predict_url = `http://127.0.0.1:5000/predictpole/${season}/${round}`;
                break;
            case 2:
                predict_url = `https://formulachaeone-495986580044.us-central1.run.app/predictquali/${season}/${round}`;
                break;
            case 3:
                predict_url = `https://formulachaeone-495986580044.us-central1.run.app/predictqualilist/${season}/${round}`;
                break;
        }
        console.log(predict_url);
        fetch(predict_url, {
            method: 'GET', referrerPolicy: "unsafe-url"
        }).then((res) => {
            console.log(res)
            if (!res.ok) {
                throw Error('Formula-Won API returned error.');
            }
            return res.text();
        }).then((resp) => {
            console.log(resp);
        }).catch((err) => {
            toast.error('Failed: ' + err.message);
        });
    }

    return (
        <div>
            {stage === 1 &&
                <div className="App">
                    <header className="App-header">
                        <p>
                            Edit <code>src/App.js</code> and save to reload.
                        </p>
                    </header>
                    <form onSubmit={search}>
                        <input onChange={e => setSeason(e.target.value)}/>
                        <input onChange={e => setRound(e.target.value)}/>
                        <button type="submit">Find</button>
                    </form>
                </div>
            }
            {stage === 2 &&
                <form onSubmit={predict}>
                    <input type="radio" value="1" onChange={e => setPrediction(e.target.value)}/>
                    <input type="radio" value="2" onChange={e => setPrediction(e.target.value)}/>
                    <input type="radio" value="3" onChange={e => setPrediction(e.target.value)}/>
                    <button type="submit">Find</button>
                </form>
            }
        </div>
    );
}

export default Home;
