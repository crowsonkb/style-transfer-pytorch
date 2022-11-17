import asyncio
from dataclasses import dataclass, is_dataclass
import io
import json
from pathlib import Path

from aiohttp import web
import torch
import torch.multiprocessing as mp
from torchvision.transforms import functional as TF

from . import srgb_profile, STIterate


@dataclass
class WIIterate:
    iterate: STIterate
    image: torch.Tensor


@dataclass
class WIDone:
    pass


@dataclass
class WIStop:
    pass


class DCJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            dct = dict(obj.__dict__)
            dct['_type'] = type(obj).__name__
            return dct
        return super().default(obj)


class WebInterface:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.q = mp.Queue()
        self.encoder = DCJSONEncoder()
        self.image = None
        self.loop = None
        self.runner = None
        self.wss = []

        self.app = web.Application()
        self.static_path = Path(__file__).resolve().parent / 'web_static'
        self.app.router.add_routes([web.get('/', self.handle_index),
                                    web.get('/image', self.handle_image),
                                    web.get('/websocket', self.handle_websocket),
                                    web.static('/', self.static_path)])

        print(f'Starting web interface at http://{self.host}:{self.port}/')
        self.process = mp.Process(target=self.run)
        self.process.start()

    async def run_app(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port, shutdown_timeout=5)
        await site.start()
        while True:
            await asyncio.sleep(3600)

    async def process_events(self):
        while True:
            f = self.loop.run_in_executor(None, self.q.get)
            await f
            event = f.result()
            if isinstance(event, WIIterate):
                self.image = event.image
                await self.send_websocket_message(event.iterate)
            elif isinstance(event, WIDone):
                await self.send_websocket_message(event)
                if self.wss:
                    print('Waiting for web clients to finish...')
                    await asyncio.sleep(5)
            elif isinstance(event, WIStop):
                for ws in self.wss:
                    await ws.close()
                if self.runner is not None:
                    await self.runner.cleanup()
                self.loop.stop()
                return

    def compress_image(self):
        buf = io.BytesIO()
        TF.to_pil_image(self.image).save(buf, format='jpeg', icc_profile=srgb_profile,
                                         quality=95, subsampling=0)
        return buf.getvalue()

    async def handle_image(self, request):
        if self.image is None:
            raise web.HTTPNotFound()
        f = self.loop.run_in_executor(None, self.compress_image)
        await f
        return web.Response(body=f.result(), content_type='image/jpeg')

    async def handle_index(self, request):
        body = (self.static_path / 'index.html').read_bytes()
        return web.Response(body=body, content_type='text/html')

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.wss.append(ws)
        async for _ in ws:
            pass
        try:
            self.wss.remove(ws)
        except ValueError:
            pass
        return ws

    async def send_websocket_message(self, msg):
        for ws in self.wss:
            try:
                await ws.send_json(msg, dumps=self.encoder.encode)
            except ConnectionError:
                try:
                    self.wss.remove(ws)
                except ValueError:
                    pass

    def put_iterate(self, iterate, image):
        self.q.put_nowait(WIIterate(iterate, image.cpu()))

    def put_done(self):
        self.q.put(WIDone())

    def close(self):
        self.q.put(WIStop())
        self.process.join(12)

    def run(self):
        self.loop = asyncio.get_event_loop()
        asyncio.ensure_future(self.run_app())
        asyncio.ensure_future(self.process_events())
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            self.q.put(WIStop())
            self.loop.run_forever()
