import inspect


# 替换一个函数
def patch_func(module, func_name):
    module.scr_func = func_name


def __new__(cls, *args, **kwargs):  # 在被替换的类传创建对象时,会自动调用这个函数
    return cls.real_cls(*args, **kwargs)  # 创建real_cls的对象，而这个real_cls就是我们自己修改后的类


# 替换一个类
def patch_class(src_class, your_class):
    setattr(src_class, 'real_cls', your_class)
    setattr(src_class, '__new__', __new__)


# 替换一个模块下的多个类
def patch_classes(module_name, class_names: []):
    assert inspect.ismodule(module_name)  # 获取模块
    class_list = {}  # 该模块中的所有类字典
    for name, class_ in inspect.getmembers(module_name, inspect.isclass):
        class_list[name] = class_  # 搜索模块中的类并添加进字典
    for class_ in class_names:
        obj = class_list[class_.__name__]  # 获取类对象对应的type
        setattr(obj, 'real_cls', class_)  # 为该class添加新的属性叫做real_cls
        setattr(obj, '__new__', __new__)  # 将创建对象的__new__函数替换为上面定义的函数
