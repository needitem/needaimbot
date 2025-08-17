#pragma once

#include <memory>
#include <string>
#include <typeinfo>
#include <functional>

namespace Core {
namespace DI {

template<typename T>
using Factory = std::function<std::unique_ptr<T>()>;

class IDependencyInjector {
public:
    virtual ~IDependencyInjector() = default;

    template<typename Interface, typename Implementation>
    void registerSingleton() {
        static_assert(std::is_base_of_v<Interface, Implementation>, 
                     "Implementation must inherit from Interface");
        
        registerSingletonImpl(
            typeid(Interface).name(),
            []() -> std::unique_ptr<void> {
                return std::make_unique<Implementation>();
            }
        );
    }

    template<typename Interface>
    void registerFactory(Factory<Interface> factory) {
        registerFactoryImpl(
            typeid(Interface).name(),
            [factory = std::move(factory)]() -> std::unique_ptr<void> {
                return factory();
            }
        );
    }

    template<typename Interface>
    std::shared_ptr<Interface> resolve() {
        auto ptr = resolveImpl(typeid(Interface).name());
        return std::static_pointer_cast<Interface>(ptr);
    }

    template<typename Interface>
    bool isRegistered() const {
        return isRegisteredImpl(typeid(Interface).name());
    }

    virtual void clear() = 0;

protected:
    virtual void registerSingletonImpl(const std::string& typeName, 
                                     std::function<std::unique_ptr<void>()> factory) = 0;
    
    virtual void registerFactoryImpl(const std::string& typeName,
                                   std::function<std::unique_ptr<void>()> factory) = 0;
    
    virtual std::shared_ptr<void> resolveImpl(const std::string& typeName) = 0;
    
    virtual bool isRegisteredImpl(const std::string& typeName) const = 0;
};

} // namespace DI
} // namespace Core