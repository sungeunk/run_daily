def get_target_pip_package_version(target_pip_package_name_list):
    # get package name and version
    import pkg_resources

    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(
        [
            f"{i.key}=={i.version}"
            for i in installed_packages
            if i.key in target_pip_package_name_list
        ]
    )

    pkg_name = ""
    pkg_version = ""
    if installed_packages_list:
        pkg_name = installed_packages_list[0].split("==")[0]
        pkg_version = installed_packages_list[0].split("==")[1]
    return pkg_name, pkg_version