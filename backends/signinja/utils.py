#!/usr/bin/env python3

import yaml
import os
from time import sleep, time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from requests import Session
from src.constants import DIR_MAIN


config_path = os.path.join(DIR_MAIN, 'backends/signinja/config.yaml')
configs = yaml.load(open(config_path), Loader=yaml.FullLoader)


def start_webdrive_instance(url, headless=True, exe=None, log_path='/dev/null'):
    o = Options()
    o.headless = headless
    if exe is None:
        driver = webdriver.Firefox(service_log_path=log_path, options=o)
    else:
        driver = webdriver.Firefox(executable_path=exe,
                                   service_log_path=log_path, options=o)
    driver.get(url)

    return driver


def headless_login(auth, url='', return_driver=False, headless=True, exe=None, log_path='/dev/null'):
    """
    To use Selenium to login and get a requests session
    that's logged in.

    :param auth: dict of list, key:[xpath, value]
                i.e. 'username': ['//input[@name="username"]', 'johndoe']
                     'password': ['//input[@name="password"]', 'abcde1234']
                     'submit': ['//input[@name="submit"]']
                also contain other params as key:value
                i.e. 'has_2fa_page': True
    :param url: login page URL
    :param return_driver: Bool, optional, whether to return
                Selenium Webdriver obj also; if not, return
                only requests Session obj
    :param headless: Bool, whether to be headless
    :param exe:
    :param log_path:
    :return: a requests session or Selenium Webdriver
    """

    if not url:
        url = auth.get('url')
        if url is None:
            raise ValueError('Need either url arg or "url" in auth')

    # navigate to login page
    if headless:
        # any site config that would require head
        if auth.get('has_2fa_page', False):
            headless = False

    driver = start_webdrive_instance(url, headless=headless, exe=exe, log_path=log_path)
    loadup_wait = int(auth.get('loadup_wait_sec', False)
                      or configs.get('loadup_wait_sec', 3))
    driver.implicitly_wait(int(loadup_wait))

    # submit authentication data
    # username
    un = auth.get('username')
    if un is not None:
        try:
            driver.find_element_by_xpath(un[0]).send_keys(un[1])
        except:
            driver.quit()
            raise
    else:
        driver.quit()
        raise ValueError('username not in auth')

    # password
    pw = auth.get('password')
    if pw is not None:
        try:
            driver.find_element_by_xpath(pw[0]).send_keys(pw[1])
        except:
            driver.quit()
            raise
    else:
        driver.quit()
        raise ValueError('password not in auth')

    # submit
    submit = auth.get('submit')
    if submit is not None:
        try:
            driver.find_element_by_xpath(submit[0]).click()

            # enforced explicit wait
            login_wait = int(auth.get('login_wait_sec', False)
                             or configs.get('login_wait_sec', 3))
            sleep(login_wait)
        except:
            driver.quit()
            raise
    else:
        driver.quit()
        raise ValueError('submit not in auth')

    if auth.get('fail_catch') is not None:
        fail_catch = auth['fail_catch']
        if 'in_page_source' in fail_catch:
            if fail_catch.get('in_page_source') in driver.page_source:
                driver.quit()
                raise ValueError('Authentication failed')

    def _login_success():
        return False
    if auth.get('login_success_catch') is not None:
        if 'in_page_source' in auth.get('login_success_catch'):
            def _login_success():
                return auth['login_success_catch']['in_page_source']\
                       in driver.page_source

    if auth.get('has_2fa_page', False):
        n0 = len(driver.page_source)
        timeout = int(auth.get('wait_2fa_timeout', False)
                      or configs.get('wait_2fa_timeout', 600))
        t0 = time()
        wait_interval = int(configs.get('wait_2fa_interval', 2))
        sleep(wait_interval)
        while (time() - t0) < timeout and \
                (not _login_success() or len(driver.page_source) == n0):
            sleep(wait_interval)

    if auth.get('login_success_catch') is not None:
        if not _login_success():
            driver.quit()
            raise ValueError('Login failed based on login_success_catch'
                             ' definition')

    # copy cookies to requests session
    session = Session()
    _ = [session.cookies.set(
        c['name'], c['value']) for c in driver.get_cookies()]

    if return_driver or auth.get('return_webdriver', False):
        return session, driver

    driver.quit()
    return session


def build_auth(xpaths, username, password):
    auth = {}

    try:
        auth['username'] = [xpaths.pop('username'), username]
    except:
        raise ValueError('username not in predefined')

    try:
        auth['password'] = [xpaths.pop('password'), password]
    except:
        raise ValueError('password not in predefined')

    try:
        auth['submit'] = [xpaths.pop('submit')]
    except:
        raise ValueError('submit not in predefined')

    return auth


def auth_from_yaml(file_path, username=None, password=None):
    auth_configs = yaml.load(open(file_path), Loader=yaml.FullLoader)
    if username is None or password is None:
        auth_file = auth_configs.get('auth_file', False)
        if auth_file and os.path.isfile(auth_file):
            username, password = open(auth_file, 'r')\
                .read().strip().split('\n')
        else:
            raise ValueError('Need either username and password'
                             ' args or valid auth_file, cannot'
                             ' find: %s' % auth_file)
    auth = build_auth(auth_configs, username, password)
    auth.update(auth_configs)
    return auth


def predefined_auth(name, username=None, password=None):
    if configs is not None and 'predefined_auths' in configs:
        d_auths = os.path.join(DIR_MAIN, configs.get('predefined_auths'))
        if not os.path.isdir(d_auths):
            raise ValueError('Predefine_auths dir does not exists')
    else:
        raise ValueError('Config missing predefine_auths dir path')

    p_auth = os.path.join(d_auths, '%s.yaml' % name)
    if not os.path.isfile(p_auth):
        raise ValueError('No such file: %s' % p_auth)

    return auth_from_yaml(p_auth, username, password)
