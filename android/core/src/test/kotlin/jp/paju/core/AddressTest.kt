package jp.paju.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/** 設定の機体の番地から、端末の口（`/device?name=…`・課題9 §6.6）へのつなぎ先を作る。 */
class AddressTest {
    @Test
    fun addsTheSchemeThePortAndTheDeviceName() {
        assertEquals("ws://100.64.0.1:8765/device?name=tablet-1", Address.url("100.64.0.1", "tablet-1"))
        assertEquals("ws://paju.local:9000/device?name=tablet-1", Address.url(" ws://paju.local:9000/ ", "tablet-1"))
        assertEquals("wss://h:8765/device?name=tablet-1", Address.url("wss://h", "tablet-1"))
    }

    @Test
    fun bracketsAnIpv6Address() {
        // Tailscale の番地は IPv6 のこともある（端-d）
        assertEquals("ws://[fd7a:115c:a1e0::1]:8765/device?name=t", Address.url("fd7a:115c:a1e0::1", "t"))
        assertEquals("ws://[fd7a::1]:9000/device?name=t", Address.url("[fd7a::1]:9000", "t"))
    }

    @Test
    fun refusesAnAddressThatIsNotAHost() {
        assertNull(Address.url("paju local", "t"))
        assertNull(Address.url("h:abc", "t"))
        assertNull(Address.url("h:99999", "t"))
        assertNull(Address.url("h/path", "t"))
    }

    @Test
    fun encodesTheDeviceName() {
        assertEquals("ws://h:8765/device?name=%E3%81%82+%26", Address.url("h", " あ & "))
    }

    @Test
    fun refusesWhatTheMachineRefuses() {
        assertNull(Address.url("", "tablet-1"))
        assertNull(Address.url("h", "  "))
        assertNull(Address.url("h", "a".repeat(33)))
        assertNull(Address.url("http://h", "tablet-1"))
    }
}
