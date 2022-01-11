<script>
    export let img_url;
    export let size = 256;
</script>
<div class="container" style="width: {size}px; height: {size}px;">
    <div class="image" style="background-image: url('{img_url}'); width: {size}px; height: {size}px;"></div>
    <div class="overlay" style="z-index: 10; width: {size}px; height: {size}px;"></div>
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
        background-color: white;
    }

    .overlay:hover {
        opacity: 0.7;
    }
</style>