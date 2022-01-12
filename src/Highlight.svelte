<script>
    import { onMount } from "svelte";

    export let img_url;
    export let size = 256;
    export let highlightRadius = 15;

    let canvas;
    let ctx;

    onMount(() => {
        canvas.width = size;
        canvas.height = size;
        ctx = canvas.getContext('2d');
    });

    function handleMove(e) {
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(e.offsetX, e.offsetY, highlightRadius, 0, 2 * Math.PI);
        ctx.fill();
    }
</script>

<div class="container" style="width: {size}px; height: {size}px;">
    <div class="image" style="background-image: url('{img_url}'); width: {size}px; height: {size}px;"></div>

    <canvas 
        bind:this={canvas}
        on:mousemove={handleMove} 
        class="overlay" 
        style="z-index: 10; width: {size}px; height: {size}px;"
    ></canvas>
</div>

<style>
    .container {
        position: relative;
        display: inline-block;
    }

    .image, .overlay {
        position: absolute;
        left: 0px;
        top: 0px;
    }

    .overlay {
        opacity: 0;
        transition: opacity .1s ease-in-out;
    }

    .overlay:hover {
        opacity: 0.7;
    }
</style>