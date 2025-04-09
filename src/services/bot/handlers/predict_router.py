import logging

import httpx
from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, Message, ReplyKeyboardRemove

import src.services.bot.handlers.keyboards as keyboards
from src.services.bot.config_reader import settings


class States(StatesGroup):
    start_enhance = State()
    upscale = State()
    upscale_x2 = State()
    upscale_x4 = State()
    deblur = State()
    denoise = State()


router = Router()


@router.message(Command("cancel"))
@router.message(F.text.lower() == "отмена")
async def stop_router(message: Message, state: FSMContext):
    """
    Cancel enhancement
    """
    await message.answer("Процесс остановлен")
    await state.clear()


@router.message(Command("enhance"))
async def begin_enhance(message: Message, state: FSMContext):
    """
    Choose type of enhancement
    """
    await message.answer(
        text="Выберите тип улучшения", reply_markup=keyboards.enhance_kb()
    )
    await state.set_state(States.start_enhance)


@router.message(States.start_enhance, F.text.lower() == "повысить разрешение")
async def choose_upscale(message: Message, state: FSMContext):
    """
    Choose upscaling
    """
    await message.answer(text="Выберите степень", reply_markup=keyboards.upscale_kb())
    await state.set_state(States.upscale)


@router.message(States.start_enhance, F.text.lower() == "убрать смазы")
async def choose_deblur(message: Message, state: FSMContext):
    """
    Choose deblurring
    """
    await message.answer(
        text="Загрузите изображение:", reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(States.deblur)


@router.message(States.start_enhance, F.text.lower() == "убрать шумы")
async def choose_denoise(message: Message, state: FSMContext):
    """
    Choose denoising
    """
    await message.answer(
        text="Загрузите изображение:", reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(States.denoise)


@router.message(States.upscale, F.text.lower() == "x2")
async def define_x2_upscale(message: Message, state: FSMContext):
    """
    Choose x2 upscale
    """
    await message.answer(
        text="Загрузите изображение:", reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(States.upscale_x2)


@router.message(States.upscale, F.text.lower() == "x4")
async def define_x4_upscale(message: Message, state: FSMContext):
    """
    Choose x4 upscale
    """
    await message.answer(
        text="Загрузите изображение:", reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(States.upscale_x4)


@router.message(States.upscale_x2, F.photo)
async def x2_upscale(message: Message, state: FSMContext, bot: Bot):
    """
    Apply x2 upscaling
    """
    file = await bot.download(message.photo[-1])
    await message.answer(text="Подождите, идёт обработка...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.BASE_SITE}/enhance/upscale",
            params={"scale": 2},
            files={"image": file.getvalue()},
            timeout=1000.0,
        )
    if response.status_code == 200:
        await message.answer(text="Готово!")
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=BufferedInputFile(file=response.content, filename="image.png"),
        )
        await state.clear()
    else:
        logging.error(f"API request error. Status code: {response.status_code}")
        await message.answer(text="Упс! Что-то пошло не так, попробуйте позже")
        await state.clear()


@router.message(States.upscale_x4, F.photo)
async def x4_upscale(message: Message, state: FSMContext, bot: Bot):
    """
    Apply x4 upscaling
    """
    file = await bot.download(message.photo[-1])
    await message.answer(text="Подождите, идёт обработка...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.BASE_SITE}/enhance/upscale",
            params={"scale": 4},
            files={"image": file.getvalue()},
            timeout=1000.0,
        )
    if response.status_code == 200:
        await message.answer(text="Готово!")
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=BufferedInputFile(file=response.content, filename="image.png"),
        )
        await state.clear()
    else:
        logging.error(f"API request error. Status code: {response.status_code}")
        await message.answer(text="Упс! Что-то пошло не так, попробуйте позже")
        await state.clear()


@router.message(States.deblur, F.photo)
async def deblur(message: Message, state: FSMContext, bot: Bot):
    """
    Apply deblurring
    """
    file = await bot.download(message.photo[-1])
    await message.answer(text="Подождите, идёт обработка...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.BASE_SITE}/enhance/deblur",
            files={"image": file.getvalue()},
            timeout=1000.0,
        )
    if response.status_code == 200:
        await message.answer(text="Готово!")
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=BufferedInputFile(file=response.content, filename="image.png"),
        )
        await state.clear()
    else:
        logging.error(f"API request error. Status code: {response.status_code}")
        await message.answer(text="Упс! Что-то пошло не так, попробуйте позже")
        await state.clear()


@router.message(States.denoise, F.photo)
async def denoise(message: Message, state: FSMContext, bot: Bot):
    """
    Apply denoising
    """
    file = await bot.download(message.photo[-1])
    await message.answer(text="Подождите, идёт обработка...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.BASE_SITE}/enhance/denoise",
            files={"image": file.getvalue()},
            timeout=1000.0,
        )
    if response.status_code == 200:
        await message.answer(text="Готово!")
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=BufferedInputFile(file=response.content, filename="image.png"),
        )
        await state.clear()
    else:
        logging.error(f"API request error. Status code: {response.status_code}")
        await message.answer(text="Упс! Что-то пошло не так, попробуйте позже")
        await state.clear()
