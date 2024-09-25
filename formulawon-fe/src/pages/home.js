import { useState } from "react";
import { toast } from "react-toastify";
import { DragDropContext, Droppable, Draggable } from "@hello-pangea/dnd";
import Backdrop from '@mui/material/Backdrop';
import CircularProgress from '@mui/material/CircularProgress';
import formulawon from '../assets/formulawon.png'
import formulawonw from '../assets/formulawonw.png'

const Home = () => {
    const [stage, setStage] = useState(1);
    const [season, setSeason] = useState(0);
    const [round, setRound] = useState(0);
    const [data, setData] = useState(null);
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);
    const [name, setName] = useState("");
    const [popup, setPopup] = useState(false);

    const get_quali_part = (item) => {
        if (item.Q3) {
            return "Q3"
        } else if (item.Q2) {
            return "Q2"
        } else {
            return "Q1";
        }
    }

    const validate = () => {
        const cur_year = new Date().getFullYear();
        if (season < 2003 || season > cur_year) {
            toast.error("ERROR: The season is invalid!");
            return false
        } else if (round < 1) {
            toast.error("ERROR: The round is invalid.");
            return false
        } else {
            return true
        }
    }

    const parse_order = () => {
        const order = Object.create(null)
        data.map((item, index) => {
            order[item.Driver.driverId] = index + 1;
        })
        return order
    }

    const search = (e) => {
        e.preventDefault();
        setLoading(true);
        if(validate()) {
            fetch(`https://ergast.com/api/f1/${season}/${round}/qualifying.json`, {
                method: 'GET', headers: {}, referrerPolicy: "unsafe-url"
            }).then((res) => {
                setLoading(false)
                if (!res.ok) {
                    throw Error('ERGAST API returned error.');
                }
                return res.json();
            }).then((resp) => {
                if (resp.MRData.RaceTable.Races === undefined || resp.MRData.RaceTable.Races.length == 0) {
                    // array does not exist or is empty
                    toast.error("ERROR: The round is invalid.");
                } else {
                    setData(resp.MRData.RaceTable.Races[0].QualifyingResults)
                    setName(resp.MRData.RaceTable.Races[0].raceName)
                    setStage(2)
                    setPopup(true);
                }
            }).catch((err) => {
                toast.error('Failed: ' + err.message);
            });
        }
        setLoading(false)
    }


    const predict = (e) => {
        e.preventDefault();
        setLoading(true);
        var predict_url = `https://formulawon-v3-495986580044.us-central1.run.app/predictqualilist/${season}/${round}`;
        fetch(predict_url, {
            method: 'GET', headers: {'order': JSON.stringify(parse_order())}
        }).then((res) => {
            setLoading(false)
            if (!res.ok) {
                throw Error('Formula-Won API returned error.');
            }
            return res.json();
        }).then((resp) => {
            setResponse(resp);
            setStage(3)
            setResponse((state) => {
                return state;
            })
        }).catch((err) => {
            toast.error('Failed: ' + err.message);
        });
    }

    function handleOnDragEnd(result) {
        if (!result.destination) return;
        const newBox = data;
        const [draggedItem] = newBox.splice(result.source.index, 1);
        newBox.splice(result.destination.index, 0, draggedItem);
        setData(newBox);
        console.log(data)
    }

    return (
        <div className="body">
            {loading &&
                <Backdrop sx={{color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1}} open = {loading}>
                    <CircularProgress size="3rem" color="secondary" />
                </Backdrop>
            }
            {popup &&
                <Backdrop sx={{color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1}} open={popup}>
                    <div className="App">
                        <div className="halo">
                            <div className="queryForm">
                                <div className="subheading">
                                    <h3>Note</h3>
                                </div>
                                <h5>Sometimes the qualifying order will not reflect the starting grid due to penalties.</h5>
                                <h5>Please use the drag and drop menu to match the order, if necessary.</h5>
                                <button className="btn" onClick={() => {
                                    setPopup(false)
                                }}>Confirm
                                </button>
                            </div>
                        </div>
                    </div>
                </Backdrop>
            }
            <div>
                {stage === 1 &&
                    <div className="App">
                        <img className="fw-logo d-inline-block align-top" src={formulawon}/>
                        <div className="halo">
                            <div className="queryForm">
                                <form onSubmit={search}>
                                    <input type="number" placeholder="SEASON" className="searchBar" onChange={e => setSeason(e.target.value)}/>
                                    <input type="number" placeholder="ROUND" className="searchBar" onChange={e => setRound(e.target.value)}/>
                                    <button className="btn" type="submit">Find</button>
                                </form>
                            </div>
                        </div>
                    </div>
                }
                {stage === 2 &&
                    <div className="scoreboard">
                        <div>
                            <img width="150" className="scoreboardlogo d-inline-block align-top" src={formulawonw}/>
                        </div>
                        <h3>{name}</h3>
                        <h5>{season}</h5>
                        <div>
                            <form onSubmit={predict}>
                                <button type="submit" className="scoreboard-btn">Find</button>
                            </form>
                        </div>
                        <DragDropContext onDragEnd={handleOnDragEnd}>
                            <Droppable droppableId="boxes">
                                {(provided) => (
                                    <div
                                        ref={provided.innerRef}
                                        {...provided.droppableProps}
                                    >
                                        {data.map((item, index) =>
                                            <Draggable key={item.number} draggableId={item.number} index={index}>
                                                {(provided) => (
                                                    <div
                                                        className="driverbox"
                                                        ref={provided.innerRef}
                                                        {...provided.dragHandleProps}
                                                        {...provided.draggableProps}
                                                    >
                                                        <div className="numberbox">
                                                            {index + 1}
                                                        </div>
                                                        <div className={'styleline'+get_quali_part(item)}>

                                                        </div>
                                                        <div className="namebox">
                                                            {item.Driver.code}
                                                        </div>
                                                        <div className="qualibox">
                                                            {get_quali_part(item)}
                                                        </div>
                                                    </div>
                                                )}
                                            </Draggable>
                                        )}
                                        {provided.placeholder}
                                    </div>
                                )}
                            </Droppable>
                        </DragDropContext>
                    </div>
                }
                {stage === 3 && typeof response.winners === 'object' &&
                    <div className="scoreboard">
                        <div>
                            <img width="150" className="scoreboardlogo d-inline-block align-top" src={formulawonw}/>
                        </div>
                        <h3>{name}</h3>
                        <h5>{season}</h5>
                        <button onClick={() => {
                            window.location.reload()
                        }} className="scoreboard-btn">Back
                        </button>
                        <h3>Most Likely</h3>
                        <div className="driverbox-nohvr">
                            <div className="numberbox">
                                {response.winners[0]}
                            </div>
                            <div className={'styleline' + get_quali_part(data[response.winners[0] - 1])}></div>
                            <div className="namebox">
                                {data[response.winners[0] - 1].Driver.code}
                            </div>
                            <div className="qualibox">
                                {get_quali_part(data[response.winners[0] - 1])}
                            </div>
                        </div>
                        <h5>Also Likely</h5>
                        <div className="driverbox-nohvr">
                            <div className="numberbox">
                                {response.winners[1]}
                            </div>
                            <div className={'styleline' + get_quali_part(data[response.winners[1] - 1])}></div>
                            <div className="namebox">
                                {data[response.winners[1] - 1].Driver.code}
                            </div>
                            <div className="qualibox">
                                {get_quali_part(data[response.winners[1] - 1])}
                            </div>
                        </div>
                        <div className="driverbox-nohvr">
                            <div className="numberbox">
                                {response.winners[2]}
                            </div>
                            <div className={'styleline' + get_quali_part(data[response.winners[2] - 1])}></div>
                            <div className="namebox">
                                {data[response.winners[2] - 1].Driver.code}
                            </div>
                            <div className="qualibox">
                                {get_quali_part(data[response.winners[2] - 1])}
                            </div>
                        </div>
                    </div>
                }
            </div>
        </div>
    );
}

export default Home;
