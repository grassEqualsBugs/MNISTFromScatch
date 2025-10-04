import { useEffect, useRef, useState } from "react";

function drawScene(canvas, drawingRadius, grid, mousePosition) {
    const context = canvas.getContext("2d");
    const size = canvas.width;
    const cellSize = size / 28;
    context.clearRect(0, 0, size, size);

    // filled squares
    for (let row = 0; row < 28; row++) {
        for (let col = 0; col < 28; col++) {
            if (grid[row][col] > 0) {
                const grayValue = 255 * (1 - grid[row][col]);
                context.fillStyle = `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
                context.fillRect(
                    col * cellSize,
                    row * cellSize,
                    cellSize,
                    cellSize,
                );
            }
        }
    }

    // grid
    context.beginPath();
    context.strokeStyle = "#ccc";
    for (let row = 0; row <= 28; row++) {
        context.moveTo(0, row * cellSize);
        context.lineTo(size, row * cellSize);
    }
    for (let col = 0; col <= 28; col++) {
        context.moveTo(col * cellSize, 0);
        context.lineTo(col * cellSize, size);
    }
    context.stroke();

    // red mouse circle
    context.beginPath();
    context.arc(
        mousePosition.x,
        mousePosition.y,
        drawingRadius,
        0,
        2 * Math.PI,
    );
    context.strokeStyle = "red";
    context.stroke();
}

function UserInput({ size, grid, setGrid }) {
    const canvasRef = useRef(null);
    const drawingRadius = (2 * size) / 28;
    const [mousePosition, setMousePosition] = useState({ x: -1, y: -1 });
    const [mouseDown, setMouseDown] = useState(false);

    // effect for handling event listeners
    useEffect(() => {
        const canvas = canvasRef.current;

        const handleMouseMove = (event) => {
            const rect = canvas.getBoundingClientRect();
            setMousePosition({
                x: event.clientX - rect.left,
                y: event.clientY - rect.top,
            });
        };
        const handleMouseDown = (event) => {
            if (event.button === 0) setMouseDown(true);
        };
        const handleMouseUp = (event) => {
            if (event.button === 0) setMouseDown(false);
        };
        canvas.addEventListener("mousemove", handleMouseMove);
        canvas.addEventListener("mousedown", handleMouseDown);
        canvas.addEventListener("mouseup", handleMouseUp);
        canvas.addEventListener("mouseleave", handleMouseUp); // also stop drawing if mouse leaves

        return () => {
            canvas.removeEventListener("mousemove", handleMouseMove);
            canvas.removeEventListener("mousedown", handleMouseDown);
            canvas.removeEventListener("mouseup", handleMouseUp);
            canvas.removeEventListener("mouseleave", handleMouseUp);
        };
    }, []);

    // effect for updating the grid when drawing
    useEffect(() => {
        if (mouseDown) {
            const cellSize = size / 28;

            // create a deep copy of the 2D grid to avoid mutation
            const newGrid = grid.map((row) => [...row]);
            let changed = false;

            for (let row = 0; row < 28; row++) {
                for (let col = 0; col < 28; col++) {
                    const cellPos = {
                        x: (col + 0.5) * cellSize,
                        y: (row + 0.5) * cellSize,
                    };
                    const dx = mousePosition.x - cellPos.x;
                    const dy = mousePosition.y - cellPos.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist <= drawingRadius) {
                        const fillValue = 1 - Math.pow(dist / drawingRadius, 4);
                        if (newGrid[row][col] < fillValue) {
                            newGrid[row][col] = fillValue;
                            changed = true;
                        }
                    }
                }
            }

            if (changed) {
                setGrid(newGrid);
            }
        }
    }, [mouseDown, drawingRadius, mousePosition, grid, setGrid, size]);

    // effect for drawing the scene
    useEffect(() => {
        const canvas = canvasRef.current;
        drawScene(canvas, drawingRadius, grid, mousePosition);
    }, [grid, drawingRadius, mousePosition]);

    return <canvas ref={canvasRef} width={size} height={size}></canvas>;
}

export default UserInput;
